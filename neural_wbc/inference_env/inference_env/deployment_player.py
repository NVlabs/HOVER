# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import argparse
import json
import os
import pprint
import torch
from typing import Any

from inference_env.neural_wbc_env import NeuralWBCEnv
from inference_env.neural_wbc_env_cfg import NeuralWBCEnvCfg

from neural_wbc.student_policy import StudentPolicyTrainer, StudentPolicyTrainerCfg


class DeploymentPlayer:
    """Player used for deploying the policy in sim or real environment.

    Used to load trained policy and use it to control robots in simulation or real environment.

    Example:
        >>> player = DeploymentPlayer(args)
        >>> player.update({"joint_positions": torch.rand(1, NUM_CONTROLS, device=device)})
        >>> player.play_once()  # single step

    """

    def __init__(
        self,
        args_cli: argparse.Namespace,
        env_cfg: NeuralWBCEnvCfg,
        custom_config: dict[str, Any] | None = None,
        demo_mode: bool = False,
    ):
        """
        Args:
            args_cli: command line arguments
            custom_config: custom configuration for the environment
            demo_mode (bool): whether to run in demo mode, without need for student policy

        Note:
            The *demo_mode* allows setting of joint manually for e.g. debugging purposes.
        """
        env_cfg.robot = args_cli.robot
        self.demo_mode = demo_mode

        if not args_cli.headless:
            env_cfg.enable_viewer = True

        if args_cli.reference_motion_path is not None:
            env_cfg.reference_motion_cfg.motion_path = args_cli.reference_motion_path

        if custom_config is not None:
            self._update_env_cfg(env_cfg=env_cfg, custom_config=custom_config)

        # Creates inference environment
        self.env = NeuralWBCEnv(cfg=env_cfg)

        assert self.env.cfg.control_type in [
            "Pos",
            "None",
        ], "Only position control or None is supported for this player."

        if not self.demo_mode:
            student_path = args_cli.student_path
            if student_path:
                with open(os.path.join(student_path, "config.json")) as fh:
                    config_dict = json.load(fh)
                config_dict["resume_path"] = student_path
                config_dict["checkpoint"] = args_cli.student_checkpoint
                student_cfg = StudentPolicyTrainerCfg(**config_dict)
                student_trainer = StudentPolicyTrainer(env=self.env, cfg=student_cfg)
                self.policy = student_trainer.get_inference_policy(device=self.env.device)
            else:
                raise ValueError("student_policy.resume_path is needed for play or eval. Please specify a value.")

    def update(self, obs_dict: dict[str, torch.Tensor]) -> None:
        """Update the kinematics of the underlying robot model based on real robot observations.

        Args:
            obs_dict (dict[str, torch.Tensor]): A dictionary containing the latest robot observations.
        """
        self.env.robot.update(obs_dict)

    def play_once(self, ext_actions: torch.Tensor | None = None):
        """Advances the environment one time step after generating observations"""
        obs = self.env.get_observations()
        with torch.inference_mode():
            if ext_actions is not None:
                actions = ext_actions
            else:
                actions = self.policy(obs)
            _, obs, dones, extras = self.env.step(actions)  # For HW, this internally just does forward
        return actions, obs, dones, extras

    def reset(self):
        """Reset the underlying environment"""
        with torch.inference_mode():
            obs, _ = self.env.reset()
            return obs

    def _update_env_cfg(self, env_cfg, custom_config: dict[str, Any]):
        """Update the default environment config if user provides a custom config. See readme for detailed usage."""
        for key, value in custom_config.items():
            obj = env_cfg
            attrs = key.split(".")
            try:
                for a in attrs[:-1]:
                    obj = getattr(obj, a)
                setattr(obj, attrs[-1], value)
            except AttributeError as atx:
                raise AttributeError(f"[ERROR]: {key} is not a valid configuration key.") from atx
        print("Updated configuration:")
        pprint.pprint(env_cfg)
