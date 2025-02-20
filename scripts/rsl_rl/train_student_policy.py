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
import json
import os
from dataclasses import MISSING, fields
from datetime import datetime

from teacher_policy_cfg import TeacherPolicyCfg
from utils import get_ppo_runner_and_checkpoint_path

from neural_wbc.student_policy import StudentPolicyTrainer, StudentPolicyTrainerCfg, TeacherPolicy

from isaaclab.app import AppLauncher

student_policy_args_prefix = StudentPolicyTrainerCfg.args_prefix()
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train student policy from a neural WBC teacher policy.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--reference_motion_path", type=str, default=None, help="Path to the reference motion dataset.")
parser.add_argument("--output_dir", type=str, default="logs", help="Directory to store the training output.")
parser.add_argument("--robot", type=str, choices=["h1", "gr1"], default="h1", help="Robot used in environment")
parser.add_argument(
    f"--{student_policy_args_prefix}root_path",
    type=str,
    default=os.path.join(project_dir, "logs/student"),
    help="Root directory of all student policy training artifacts.",
)
parser.add_argument(
    f"--{student_policy_args_prefix}append_timestamp",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If true the current timestamp will be added to the output student policy path.",
)
# append StudentPolicyTrainerCfg arguments
StudentPolicyTrainerCfg.add_args_to_parser(parser)
# append RSL-RL cli arguments
TeacherPolicyCfg.add_args_to_parser(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

# Import extensions to set up environment tasks
from vecenv_wrapper import RslRlNeuralWBCVecEnvWrapper

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class NeuralWBCTeacherPolicy(TeacherPolicy):
    """Class to load and apply a policy trained with neural WBC."""

    def __init__(self, env: RslRlNeuralWBCVecEnvWrapper, teacher_policy_cfg: TeacherPolicyCfg):
        self.path = teacher_policy_cfg.runner.resume_path
        # Here we assume that the path to the teacher policy is {experiment_dir}/{teacher_name}.
        self.name = self.path.strip("/").split("/")[-1]

        self._ppo_runner, checkpoint_path = get_ppo_runner_and_checkpoint_path(
            teacher_policy_cfg=teacher_policy_cfg, wrapped_env=env, device=env.device
        )
        self._ppo_runner.load(checkpoint_path)
        print(f"[INFO]: Loaded model checkpoint from: {checkpoint_path}")

        self._actor_critic = self._ppo_runner.alg.actor_critic
        self._actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if env.device is not None:
            self._actor_critic.to(env.device)

    def act_rollout(self, observations):
        # use act from actor_critic
        return self._actor_critic.act(observations)

    def act(self, observations):
        return self._actor_critic.act_inference(observations)


def resolve_student_policy_path(root_path: str, teacher_policy: NeuralWBCTeacherPolicy, append_timestamp: bool):
    """
    Generates a path for storing student policy configurations based on the root path and teacher policy.

    This function appends a timestamp and the teacher policy name to the given root path to create a unique path
    for storing student policy configurations.

    Args:
        root_path (str): The root directory path where the student policy configurations will be stored.
        teacher_policy (NeuralWBCTeacherPolicy): The teacher policy instance whose name will be used in the generated path.

    Returns:
        str: The generated absolute path for storing the student policy configurations.
    """
    path = os.path.join(os.path.abspath(root_path), teacher_policy.name)
    if append_timestamp:
        path += "_" + datetime.now().strftime("%y_%m_%d_%H-%M-%S")

    return path


def get_config_values_from_argparser(args: argparse.Namespace, teacher_policy: NeuralWBCTeacherPolicy):
    """
    Extracts configuration values from command-line arguments based on a predefined prefix and field names.

    This function processes the command-line arguments, filters out those that match a specified prefix,
    and returns a dictionary of configuration values that are relevant to the StudentPolicyTrainerCfg dataclass.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        dict: A dictionary containing configuration values extracted from the command-line arguments.
    """
    # Get the set of all field names in the StudentPolicyTrainerCfg dataclass
    field_names = {field_info.name for field_info in fields(StudentPolicyTrainerCfg)}

    # Extract arguments from the args object and remove the prefix
    args_dict = {
        key.removeprefix(student_policy_args_prefix): value
        for key, value in vars(args).items()
        if key.startswith(student_policy_args_prefix) and key.removeprefix(student_policy_args_prefix) in field_names
    }

    args_dict["teacher_policy"] = teacher_policy

    # Set student policy path
    args_dict["student_policy_path"] = resolve_student_policy_path(
        root_path=getattr(args, f"{student_policy_args_prefix}root_path"),
        teacher_policy=teacher_policy,
        append_timestamp=getattr(args, f"{student_policy_args_prefix}append_timestamp"),
    )

    return args_dict


def get_student_trainer_cfg(args: argparse.Namespace, env: NeuralWBCEnv, teacher_policy: NeuralWBCTeacherPolicy):
    """
    Create an instance of StudentPolicyTrainerCfg from command-line arguments, environment configuration, and a teacher policy.

    This function processes the command-line arguments, extracts necessary values, fills in any missing values using
    the environment configuration, and creates an instance of the StudentPolicyTrainerCfg dataclass.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        env (NeuralWBCEnv): The environment object that provides necessary configuration values.
        teacher_policy (NeuralWBCTeacherPolicy): The teacher policy instance to be included in the configuration.

    Returns:
        StudentPolicyTrainerCfg: An instance of the StudentPolicyTrainerCfg dataclass.

    Raises:
        ValueError: If a required field does not have a default value and is not provided in the arguments or the environment configuration.
    """
    # First try loading pre-existing config.
    previous_config = load_student_policy_trainer_cfg(args=args, teacher_policy=teacher_policy)
    if previous_config:
        return previous_config

    args_dict = get_config_values_from_argparser(args=args, teacher_policy=teacher_policy)

    # Identify fields that do not have default values and are not provided in the arguments
    no_default_fields = [
        field_info.name
        for field_info in fields(StudentPolicyTrainerCfg)
        if args_dict.get(field_info.name) is None
        and field_info.default is MISSING
        and field_info.default_factory is MISSING
    ]

    # Fill in missing values for fields that do not have defaults
    for name in no_default_fields:
        value = None
        if name == "num_policy_obs":
            # Special case: get the value from the environment's num_observations attribute
            value = env.num_observations
        elif hasattr(env, name):
            # Check if the environment has an attribute matching the field name
            value = getattr(env, name)
        elif hasattr(env.cfg, name):
            # Check if the environment's configuration has an attribute matching the field name
            value = getattr(env.cfg, name)
        else:
            # Raise an error if the field value cannot be determined
            raise ValueError(
                f"student_policy.{name} does not have a default value in the student training configuration or the"
                " environment. Please specify a value."
            )
        # Update the args_dict with the determined value
        args_dict[name] = value

    return StudentPolicyTrainerCfg(**args_dict)


def load_student_policy_trainer_cfg(args: argparse.Namespace, teacher_policy: NeuralWBCTeacherPolicy):
    """
    Loads the student policy trainer configuration from a specified resume path.

    This function checks if a resume path is provided in the command-line arguments. If so, it loads the configuration
    from a config.json at the resume path, updates it with any new arguments provided, and returns an instance of
    StudentPolicyTrainerCfg.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        teacher_policy (NeuralWBCTeacherPolicy): The teacher policy instance to be included in the configuration.

    Returns:
        StudentPolicyTrainerCfg or None: An instance of the StudentPolicyTrainerCfg dataclass if a resume path is provided,
                                         otherwise None.
    """
    # Check if resume_path is specified by the user.
    resume_path = getattr(args, f"{student_policy_args_prefix}resume_path")
    if not resume_path:
        return None

    with open(os.path.join(resume_path, "config.json")) as fh:
        config_dict = json.load(fh)

    del config_dict["teacher_policy"]

    args_dict = get_config_values_from_argparser(args=args, teacher_policy=teacher_policy)
    for key, value in args_dict.items():
        # Only overwrite values from arguments that are not None.
        if value is not None:
            config_dict[key] = value

    return StudentPolicyTrainerCfg(**config_dict)


def main():
    # parse configuration
    if args_cli.robot == "h1":
        env_cfg = NeuralWBCEnvCfgH1(mode=NeuralWBCModes.DISTILL)
    elif args_cli.robot == "gr1":
        raise ValueError("GR1 is not yet implemented")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 20
    env_cfg.terrain.env_spacing = 20
    if args_cli.reference_motion_path:
        env_cfg.reference_motion_manager.motion_path = args_cli.reference_motion_path

    # Create env and wrap it for RSL RL.
    env = NeuralWBCEnv(cfg=env_cfg)
    wrapped_env = RslRlNeuralWBCVecEnvWrapper(env)

    teacher_policy_cfg = TeacherPolicyCfg.from_argparse_args(args_cli)
    teacher_policy = NeuralWBCTeacherPolicy(env=wrapped_env, teacher_policy_cfg=teacher_policy_cfg)

    # set seed of the environment
    env.seed(teacher_policy_cfg.seed)

    trainer_cfg = get_student_trainer_cfg(args=args_cli, env=env, teacher_policy=teacher_policy)
    os.makedirs(trainer_cfg.student_policy_path, exist_ok=True)
    env_cfg.save(os.path.join(trainer_cfg.student_policy_path, "env_config.json"))
    trainer = StudentPolicyTrainer(env=wrapped_env, cfg=trainer_cfg)
    trainer.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
