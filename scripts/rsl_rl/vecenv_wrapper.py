# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""


import gymnasium as gym
import torch

from neural_wbc.core import EnvironmentWrapper
from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv


class RslRlNeuralWBCVecEnvWrapper(EnvironmentWrapper):
    """Wraps around Isaac Lab environment for RSL RL library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "critic" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: NeuralWBCEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
        """
        super().__init__(mode=env.cfg.mode)

        # initialize the wrapper
        self._env = env
        # store information required by wrapper
        self.num_envs = self._env.num_envs
        self.device = self._env.device
        self.reference_motion_manager = self._env.reference_motion_manager
        self.max_episode_length = self._env.max_episode_length
        if hasattr(self._env, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = self._env.action_space.shape[1]
        if hasattr(self._env, "observation_manager"):
            self.num_obs = self._env.observation_manager.group_obs_dim["teacher_policy"][0]
        else:
            self.num_obs = self._env.num_observations
        # -- privileged observations
        if hasattr(self._env, "observation_manager") and "critic" in self._env.observation_manager.group_obs_dim:
            self.num_privileged_obs = self._env.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self._env, "state_space"):
            self.num_privileged_obs = self._env.state_space.shape[1]
        else:
            self.num_privileged_obs = 0
        # reset at the start since the RSL-RL runner does not call reset
        self._env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self._env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self._env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self._env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self._env

    """
    Properties
    """

    def get_full_observations(self):
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
            if self.unwrapped.cfg.observation_noise_model:
                obs_dict["teacher_policy"] = self.unwrapped._observation_noise_model.apply(obs_dict["teacher_policy"])
        return obs_dict

    def get_teacher_observations(self) -> torch.Tensor:
        """Returns the current observations of the environment."""
        return self.get_full_observations()["teacher_policy"]

    def get_privileged_observations(self) -> torch.Tensor:
        """Returns the current observations of the environment."""
        return self.get_full_observations()["critic"]

    def get_student_observations(self) -> torch.Tensor:
        return self.get_full_observations()["student_policy"]

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self._env.reset()
        # return observations
        if self._mode.is_distill_mode():
            return obs_dict["student_policy"], torch.tensor([])
        return obs_dict["teacher_policy"], obs_dict["critic"]

    def step(self, actions: torch.Tensor):
        # record step information
        obs_dict, rew, terminated, truncated, extras = self._env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["student_policy"] if self._mode.is_distill_mode() else obs_dict["teacher_policy"]
        privileged_obs = obs_dict["critic"]
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        extras["time_outs"] = truncated

        # return the step information
        return obs, privileged_obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self._env.close()
