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


import torch
import unittest

from .test_main import APP_IS_READY

if APP_IS_READY:
    from neural_wbc.core.modes import NeuralWBCModes
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1


class TestNeuralWBCEnv(unittest.TestCase):
    def _check_observations(self, env: NeuralWBCEnv, obs: dict[str, torch.Tensor]):
        for key, value in obs.items():
            # Make sure that all observations do not contain NaN values.
            self.assertFalse(torch.any(torch.isnan(value)), msg=f"Observation {key} has NaN values {value}.")
        # Ensure that the policy and critic observations have shapes matching the configuration.
        self.assertEqual(obs["teacher_policy"].shape, env.observation_space.shape)
        self.assertEqual(obs["critic"].shape, env.state_space.shape)

    def test_step(self):
        # create environment configuration
        env_cfg = NeuralWBCEnvCfgH1()
        env_cfg.scene.num_envs = 100
        env_cfg.scene.env_spacing = 20
        env_cfg.mode = NeuralWBCModes.DISTILL  # Set modes to distill so that all observations are tested.

        # number of episodes to collect
        num_demos = 0

        # setup RL environment
        env = NeuralWBCEnv(cfg=env_cfg)
        env.reset()

        # sample actions
        joint_efforts = torch.zeros((env.num_envs, env.num_actions))

        # simulate physics
        while num_demos < 10 * env_cfg.scene.num_envs:
            with torch.inference_mode():

                # store signals
                # -- obs
                obs = env._get_observations()
                self._check_observations(env=env, obs=obs)

                # step the environment
                obs, _, terminated, truncated, _ = env.step(joint_efforts)
                num_demos += len(terminated) + len(truncated)
                self._check_observations(env=env, obs=obs)
