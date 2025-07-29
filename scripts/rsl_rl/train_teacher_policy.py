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


"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

# local imports
from teacher_policy_cfg import TeacherPolicyCfg

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with RSL-RL.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--reference_motion_path", type=str, default=None, help="Path to the reference motion dataset.")
parser.add_argument("--robot", type=str, choices=["h1", "gr1"], default="h1", help="Robot used in environment")

# append RSL-RL cli arguments
TeacherPolicyCfg.add_args_to_parser(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from utils import get_customized_rsl_rl, get_ppo_runner_and_checkpoint_path

import os
import torch
from datetime import datetime

# Import your specific module/class
get_customized_rsl_rl()
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

# Import extensions to set up environment tasks
from vecenv_wrapper import RslRlNeuralWBCVecEnvWrapper

from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with RSL-RL agent."""
    # parse configuration
    if args_cli.robot == "h1":
        env_cfg = NeuralWBCEnvCfgH1()
    elif args_cli.robot == "gr1":
        raise ValueError("GR1 is not yet implemented")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 20
    env_cfg.terrain.env_spacing = 20
    if args_cli.reference_motion_path:
        env_cfg.reference_motion_manager.motion_path = args_cli.reference_motion_path

    teacher_policy_cfg = TeacherPolicyCfg.from_argparse_args(args_cli)

    # Create env and wrap it for RSL RL.
    env = NeuralWBCEnv(cfg=env_cfg)
    env = RslRlNeuralWBCVecEnvWrapper(env)

    # specify directory for logging experiments
    log_dir = os.path.join(
        os.path.abspath(teacher_policy_cfg.runner.path), datetime.now().strftime("%y_%m_%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Log directory: {log_dir}")

    if teacher_policy_cfg.runner.resume:
        ppo_runner, checkpoint_path = get_ppo_runner_and_checkpoint_path(
            teacher_policy_cfg=teacher_policy_cfg, wrapped_env=env, device=env.unwrapped.device, log_dir=log_dir
        )
        ppo_runner.load(checkpoint_path)
        print(f"[INFO]: Loaded model checkpoint from: {checkpoint_path}")
    else:
        ppo_runner = OnPolicyRunner(env, teacher_policy_cfg.to_dict(), log_dir=log_dir, device=env.unwrapped.device)

    # Store the configuration
    teacher_policy_cfg.save(os.path.join(log_dir, "config.json"))
    env_cfg.save(os.path.join(log_dir, "env_config.json"))

    # set seed of the environment
    env.seed(teacher_policy_cfg.seed)

    # run training
    ppo_runner.learn(num_learning_iterations=teacher_policy_cfg.runner.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
