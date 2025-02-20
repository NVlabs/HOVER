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


import argparse
import json
import os
import pprint
import torch
from typing import Any

from isaaclab_rl.rsl_rl import export_policy_as_onnx
from teacher_policy_cfg import TeacherPolicyCfg
from utils import get_ppo_runner_and_checkpoint_path
from vecenv_wrapper import RslRlNeuralWBCVecEnvWrapper

from neural_wbc.core.evaluator import Evaluator
from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1
from neural_wbc.student_policy import StudentPolicyTrainer, StudentPolicyTrainerCfg


class Player:
    """Base class of a policy player."""

    def __init__(self, args_cli: argparse.Namespace, randomize: bool, custom_config: dict[str, Any] | None):
        # parse configuration
        mode = NeuralWBCModes.TRAIN if randomize else NeuralWBCModes.TEST
        self.student_player = False
        if args_cli.student_player:
            mode = NeuralWBCModes.DISTILL if randomize else NeuralWBCModes.DISTILL_TEST
            self.student_player = True
        if args_cli.robot == "h1":
            env_cfg = NeuralWBCEnvCfgH1(mode=mode)
        elif args_cli.robot == "gr1":
            raise ValueError("GR1 is not yet implemented")
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.scene.env_spacing = args_cli.env_spacing
        env_cfg.terrain.env_spacing = args_cli.env_spacing
        if custom_config is not None:
            self._update_env_cfg(env_cfg=env_cfg, custom_config=custom_config)
        if args_cli.reference_motion_path:
            env_cfg.reference_motion_manager.motion_path = args_cli.reference_motion_path

        # Create environment and wrap it for RSL RL.
        self.env = NeuralWBCEnv(cfg=env_cfg)
        self.wrapped_env = RslRlNeuralWBCVecEnvWrapper(self.env)

        if self.student_player:
            student_path = args_cli.student_path
            if student_path:
                with open(os.path.join(student_path, "config.json")) as fh:
                    config_dict = json.load(fh)
                config_dict["teacher_policy"] = None
                config_dict["resume_path"] = student_path
                config_dict["checkpoint"] = args_cli.student_checkpoint
                student_cfg = StudentPolicyTrainerCfg(**config_dict)
                student_trainer = StudentPolicyTrainer(env=self.wrapped_env, cfg=student_cfg)
                self.policy = student_trainer.get_inference_policy(device=self.env.device)
            else:
                raise ValueError("student_policy.resume_path is needed for play or eval. Please specify a value.")
        else:
            teacher_policy_cfg = TeacherPolicyCfg.from_argparse_args(args_cli)
            ppo_runner, checkpoint_path = get_ppo_runner_and_checkpoint_path(
                teacher_policy_cfg=teacher_policy_cfg, wrapped_env=self.wrapped_env, device=self.env.device
            )
            ppo_runner.load(checkpoint_path)
            print(f"[INFO]: Loaded model checkpoint from: {checkpoint_path}")

            # obtain the trained policy for inference
            self.policy = ppo_runner.get_inference_policy(device=self.env.device)

            # export policy to onnx
            export_model_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
            export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    def _update_env_cfg(self, env_cfg: NeuralWBCEnvCfgH1, custom_config: dict[str, Any]):
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

    def play(self, simulation_app):
        obs = self.wrapped_env.get_observations()

        # simulate environment
        while simulation_app.is_running() and not self._should_stop():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = self.policy(obs)
                # env stepping
                obs, privileged_obs, rewards, dones, extras = self.wrapped_env.step(actions)
                obs = self._post_step(
                    obs=obs, privileged_obs=privileged_obs, rewards=rewards, dones=dones, extras=extras
                )

        # close the simulator
        self.env.close()

    def _should_stop(self):
        return NotImplemented

    def _post_step(
        self, obs: torch.Tensor, privileged_obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ):
        return NotImplemented


class DemoPlayer(Player):
    """The demo player plays policy until a KeyboardInterrupt exception occurs."""

    def __init__(self, args_cli: argparse.Namespace, randomize: bool):
        super().__init__(args_cli=args_cli, randomize=randomize, custom_config=None)

    def _should_stop(self):
        return False

    def _post_step(
        self, obs: torch.Tensor, privileged_obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ):
        return obs


class EvaluationPlayer(Player):
    """The evaluation player iterates through a reference motion dataset and collects metrics."""

    def __init__(
        self, args_cli: argparse.Namespace, metrics_path: str | None = None, custom_config: dict[str, Any] | None = None
    ):
        super().__init__(randomize=False, args_cli=args_cli, custom_config=custom_config)
        self._evaluator = Evaluator(env_wrapper=self.wrapped_env, metrics_path=metrics_path)

    def play(self, simulation_app):
        super().play(simulation_app=simulation_app)
        self._evaluator.conclude()

    def _should_stop(self):
        return self._evaluator.is_evaluation_complete()

    def _post_step(
        self, obs: torch.Tensor, privileged_obs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ):
        reset_env = self._evaluator.collect(dones=dones, info=extras)
        if reset_env and not self._evaluator.is_evaluation_complete():
            self._evaluator.forward_motion_samples()
            obs, _ = self.wrapped_env.reset()

        return obs
