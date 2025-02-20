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
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import List, Type

from dataclass_wizard import fromdict


@dataclass
class PolicyCfg:
    """Configuration settings for the policy."""

    ACTIVATION_TYPES = ["elu", "selu", "relu", "crelu", "lrelu", "tanh", "sigmoid"]

    init_noise_std: float = field(default=1.0, metadata={"description": "Initial noise standard deviation for policy."})
    actor_hidden_dims: List[int] = field(
        default_factory=lambda: [512, 256, 128], metadata={"description": "Hidden dimensions for the actor network."}
    )
    critic_hidden_dims: List[int] = field(
        default_factory=lambda: [512, 256, 128], metadata={"description": "Hidden dimensions for the critic network."}
    )
    activation: str = field(default="elu", metadata={"description": "Activation function for the neural networks."})

    def __post_init__(self):
        if self.activation not in self.ACTIVATION_TYPES:
            raise ValueError(
                f"Invalid policy activation {self.activation}. Allowed values are {self.ACTIVATION_TYPES}."
            )


@dataclass
class AlgorithmCfg:
    """Configuration settings for the algorithm."""

    value_loss_coef: float = field(default=1.0, metadata={"description": "Coefficient for value loss."})
    use_clipped_value_loss: bool = field(default=True, metadata={"description": "Whether to use clipped value loss."})
    clip_param: float = field(default=0.2, metadata={"description": "Clipping parameter for PPO."})
    entropy_coef: float = field(default=0.005, metadata={"description": "Coefficient for entropy regularization."})
    num_learning_epochs: int = field(default=5, metadata={"description": "Number of learning epochs per update."})
    num_mini_batches: int = field(default=4, metadata={"description": "Number of mini-batches per epoch."})
    learning_rate: float = field(default=1.0e-3, metadata={"description": "Learning rate for the optimizer."})
    schedule: str = field(default="adaptive", metadata={"description": "Learning rate schedule type."})
    gamma: float = field(default=0.99, metadata={"description": "Discount factor for rewards."})
    lam: float = field(default=0.95, metadata={"description": "Lambda for Generalized Advantage Estimation."})
    desired_kl: float = field(default=0.01, metadata={"description": "Desired KL divergence for policy updates."})
    max_grad_norm: float = field(default=0.2, metadata={"description": "Maximum norm for gradient clipping."})


@dataclass
class RunnerCfg:
    """Configuration settings for the runner."""

    path: str = field(default="logs/teacher", metadata={"description": "Directory to save policy and logs to"})
    policy_class_name: str = field(default="ActorCritic", metadata={"description": "Class name for the policy."})
    algorithm_class_name: str = field(default="PPO", metadata={"description": "Class name for the algorithm."})
    num_steps_per_env: int = field(
        default=24, metadata={"description": "Number of steps per environment per iteration."}
    )
    max_iterations: int = field(default=10000000, metadata={"description": "Maximum number of policy updates."})
    save_interval: int = field(default=500, metadata={"description": "Interval for saving checkpoints."})
    checkpoint: str = field(default="", metadata={"description": "Checkpoint to load."})
    resume: bool = field(default=False, metadata={"description": "Whether to resume from a checkpoint."})
    resume_path: str = field(default="", metadata={"description": "Path to the checkpoint for resuming."})

    @staticmethod
    def args_prefix():
        """
        Get the prefix to be used for command-line arguments.

        Returns:
            str: The prefix string for command-line arguments.
        """
        return "runner."


@dataclass
class TeacherPolicyCfg:
    """Main configuration class that aggregates all required configurations for teacher policy training."""

    seed: int = field(default=1, metadata={"description": "Random seed for reproducibility."})
    policy: PolicyCfg = field(default_factory=PolicyCfg, metadata={"description": "Policy configuration settings."})
    algorithm: AlgorithmCfg = field(
        default_factory=AlgorithmCfg, metadata={"description": "Algorithm configuration settings."}
    )
    runner: RunnerCfg = field(default_factory=RunnerCfg, metadata={"description": "Runner configuration settings."})

    def to_dict(self):
        return asdict(self)

    def save(self, filepath: str):
        """Saves the configuration to a YAML file."""
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file)

    @staticmethod
    def load(filepath: str) -> "TeacherPolicyCfg":
        """Loads the configuration from a YAML file."""
        with open(filepath, encoding="utf-8") as file:
            data = json.load(file)
        cfg = fromdict(TeacherPolicyCfg, data)
        return cfg

    @staticmethod
    def args_prefix():
        """
        Get the prefix to be used for command-line arguments.

        Returns:
            str: The prefix string for command-line arguments.
        """
        return "teacher_policy."

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser, default_overwrites: dict = {}):
        """Adds configuration fields to an ArgumentParser."""
        group = parser.add_argument_group("Teacher policy configurations (RSL RL)")

        def add_fields_to_parser(fields):
            for field_info in fields:
                arg_name = f"--{TeacherPolicyCfg.args_prefix()}{field_info.name}"
                arg_type = field_info.type
                arg_help = field_info.metadata.get("description", "")
                default = None

                if field_info.name in default_overwrites:
                    default = default_overwrites[field_info.name]
                elif field_info.default is not MISSING:
                    default = field_info.default
                elif field_info.default_factory is not MISSING:
                    default = field_info.default_factory()

                if isinstance(default, list):  # Special handling for lists
                    group.add_argument(arg_name, type=type(default[0]), nargs="+", default=default, help=arg_help)
                elif isinstance(default, bool):
                    if default:
                        group.add_argument(arg_name, action="store_false", help=arg_help)
                    else:
                        group.add_argument(arg_name, action="store_true", help=arg_help)
                else:
                    group.add_argument(arg_name, type=arg_type, default=default, help=arg_help)

        # Add fields from each section with a prefix for context
        add_fields_to_parser(f for f in fields(TeacherPolicyCfg) if not is_dataclass(f.type))
        add_fields_to_parser(fields(PolicyCfg))
        add_fields_to_parser(fields(AlgorithmCfg))
        add_fields_to_parser(fields(RunnerCfg))

    @staticmethod
    def from_argparse_args(args: argparse.Namespace) -> "TeacherPolicyCfg":
        """Creates an instance from argparse arguments."""

        def extract_fields(cls: Type) -> dict:
            """Helper function to extract fields for a given dataclass type."""
            extracted_fields = {
                field.name: getattr(args, TeacherPolicyCfg.args_prefix() + field.name)
                for field in fields(cls)
                if hasattr(args, TeacherPolicyCfg.args_prefix() + field.name)
            }
            return extracted_fields

        policy_args = extract_fields(PolicyCfg)
        algorithm_args = extract_fields(AlgorithmCfg)
        runner_args = extract_fields(RunnerCfg)

        return TeacherPolicyCfg(
            seed=getattr(args, f"{TeacherPolicyCfg.args_prefix()}seed"),
            policy=PolicyCfg(**policy_args),
            algorithm=AlgorithmCfg(**algorithm_args),
            runner=RunnerCfg(**runner_args),
        )

    def overwrite_policy_cfg_from_file(self, file_path: str):
        try:
            saved_config = TeacherPolicyCfg.load(file_path)
            self.policy = saved_config.policy
        except FileNotFoundError:
            pass
