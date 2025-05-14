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


"""Launch Isaac Sim Simulator first."""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train student policy from a neural WBC teacher policy.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--reference_motion_path",
    type=str,
    default=None,
    help="Path to the reference motion dataset.",
)
parser.add_argument(
    "--teacher_load_run",
    type=str,
    default=".*",
    help="Experiment name of the teacher. Default is '.*' which fetches the latest experiment in the directory",
)
parser.add_argument(
    "--teacher_load_checkpoint",
    type=str,
    default="model_.*.pt",
    help="Checkpoint of the teacher. Default is 'model_.*.pt' which fetches the latest checkpoint in the run directory",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs",
    help="Directory to store the training output.",
)
parser.add_argument(
    "--robot",
    type=str,
    choices=["h1", "gr1"],
    default="h1",
    help="Robot used in environment",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from rsl_rl.runners import OnPolicyRunner
from student_policy_cfg import StudentPolicyRunnerCfg

# Import extensions to set up environment tasks
from vecenv_wrapper import RslRlNeuralWBCVecEnvWrapper

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1

from isaaclab_tasks.utils import get_checkpoint_path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # parse configuration
    if args_cli.robot == "h1":
        env_cfg = NeuralWBCEnvCfgH1(mode=NeuralWBCModes.DISTILL)
    elif args_cli.robot == "gr1":
        raise ValueError("GR1 is not yet implemented")
    agent_cfg = cli_args.update_rsl_rl_cfg(StudentPolicyRunnerCfg(), args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 20
    env_cfg.terrain.env_spacing = 20
    if args_cli.reference_motion_path:
        env_cfg.reference_motion_manager.motion_path = args_cli.reference_motion_path

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Create env and wrap it for RSL RL.
    env = NeuralWBCEnv(cfg=env_cfg)
    env = RslRlNeuralWBCVecEnvWrapper(env)

    # get the log path of the teacher policy
    log_root_path = os.path.join("logs", "rsl_rl", args_cli.robot, agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    teacher_root_path = os.path.join("logs", "rsl_rl", args_cli.robot, "teacher_policy")
    teacher_root_path = os.path.abspath(teacher_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"[INFO] Loading teacher policy from: {teacher_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    trained_teacher_path = get_checkpoint_path(
        teacher_root_path, args_cli.teacher_load_run, args_cli.teacher_load_checkpoint
    )

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    distillation_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.unwrapped.device)

    distillation_runner.load(trained_teacher_path)
    if agent_cfg.resume:
        distillation_runner.load(resume_path)

    distillation_runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
