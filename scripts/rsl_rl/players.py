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
import os
import pprint
import torch
from typing import Any

import cli_args
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner
from student_policy_cfg import StudentPolicyRunnerCfg
from teacher_policy_cfg import TeacherPolicyRunnerCfg
from vecenv_wrapper import RslRlNeuralWBCVecEnvWrapper

from neural_wbc.core.evaluator import Evaluator
from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1

from isaaclab_tasks.utils import get_checkpoint_path


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
        self.env = RslRlNeuralWBCVecEnvWrapper(self.env)

        runner_cfg = TeacherPolicyRunnerCfg() if not self.student_player else StudentPolicyRunnerCfg()
        agent_cfg = cli_args.update_rsl_rl_cfg(runner_cfg, args_cli)

        log_root_path = os.path.join("logs", "rsl_rl", args_cli.robot, agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)

        # load the policy from checkpoint
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        ppo_runner = OnPolicyRunner(env=self.env, train_cfg=agent_cfg.to_dict(), log_dir=None)
        ppo_runner.load(resume_path)
        print(f"[INFO]: Loaded model checkpoint from: {resume_path}")

        # obtain the trained policy for inference
        self.policy = ppo_runner.get_inference_policy(device=self.env.device)

        # export policy to onnx
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        export_policy_as_onnx(
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )

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
        obs, extras = self.env.get_observations()

        # simulate environment
        while simulation_app.is_running() and not self._should_stop():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = self.policy(obs)
                # env stepping
                obs, rewards, dones, extras = self.env.step(actions)
                obs = self._post_step(
                    obs=obs,
                    privileged_obs=extras["observations"]["critic"],
                    rewards=rewards,
                    dones=dones,
                    extras=extras,
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
        self._evaluator = Evaluator(env_wrapper=self.env, metrics_path=metrics_path)

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
            obs, _ = self.env.reset()

        return obs
