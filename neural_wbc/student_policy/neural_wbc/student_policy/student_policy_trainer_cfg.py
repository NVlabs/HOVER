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
from dataclasses import MISSING, dataclass, field, fields
from typing import List

from neural_wbc.student_policy.teacher_policy import TeacherPolicy


@dataclass
class StudentPolicyTrainerCfg:
    teacher_policy: TeacherPolicy = field(metadata={"description": "Path to the teacher policy model"})
    student_policy_path: str = field(metadata={"description": "Directory to save the student policy and logs to"})

    # Env
    num_policy_obs: int = field(metadata={"description": "Number of policy observations"})
    num_student_obs: int = field(metadata={"description": "Number of student observations"})
    num_actions: int = field(metadata={"description": "Number of actions"})

    # Training
    resume_path: str = field(
        default="", metadata={"description": "Directory to load a checkpoint to resume training or testing"}
    )
    checkpoint: str = field(default="", metadata={"description": "Checkpoint to resume training from"})
    num_steps_per_env: int = field(
        default=1, metadata={"description": "Maximum steps for one rollout during student training"}
    )
    max_iteration: int = field(default=200000, metadata={"description": "Maximum training iteration"})
    save_iteration: int = field(
        default=500, metadata={"description": "How many iterations between two saved checkpoints"}
    )
    student_rollout_iteration: int = field(
        default=0, metadata={"description": "After how many iterations we begin to use student policy to rollout"}
    )
    learning_rate: float = field(default=5e-4, metadata={"description": "Learning rate"})
    max_grad_norm: float = field(default=0.2, metadata={"description": "Maximum gradient norm"})
    num_mini_batches: int = field(default=1, metadata={"description": "Number of mini-batches"})
    num_learning_epochs: int = field(default=1, metadata={"description": "Number of learning epochs"})
    dagger_coefficient: float = field(default=1.0, metadata={"description": "DAgger coefficient"})

    # Policy net
    policy_hidden_dims: List[int] = field(
        default_factory=lambda: [512, 256, 128], metadata={"description": "Policy hidden dimensions"}
    )
    activation: str = field(default="elu", metadata={"description": "Activation function"})
    noise_std: float = field(default=0.001, metadata={"description": "Noise standard deviation"})

    def save(self, file_path: str):
        """
        Save the dataclass fields to a JSON file.

        Args:
            file_path (str): The path to the file where the JSON will be saved.
        """
        # Convert the dataclass to a dictionary
        data_dict = {
            field_info.name: getattr(self, field_info.name)
            for field_info in fields(self)
            if field_info.name != "teacher_policy"
        }

        # Customize the teacher_policy field in the dictionary
        teacher_policy_path = self.teacher_policy.path
        if teacher_policy_path is not None:
            data_dict["teacher_policy"] = teacher_policy_path

        # Write the dictionary to a JSON file
        with open(file_path, "w") as f:
            json.dump(data_dict, f, indent=4)

    @staticmethod
    def args_prefix():
        """
        Get the prefix to be used for command-line arguments.

        Returns:
            str: The prefix string for command-line arguments.
        """
        return "student_policy."

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser, default_overwrites: dict = {}):
        """
        Add the fields of the dataclass (except for `teacher_policy`) to an ArgumentParser.

        This method iterates over the fields of the StudentPolicyTrainerCfg dataclass and adds them as arguments
        to the provided ArgumentParser. The `teacher_policy` field is skipped. If a field has a default value or
        a default factory, that value is used as the default for the argument. The `default_overwrites` dictionary
        can be used to override the default values for specific fields.

        Args:
            parser (argparse.ArgumentParser): The argument parser to which the arguments will be added.
            default_overwrites (dict): A dictionary of field names and their corresponding default values to overwrite.
        """
        group = parser.add_argument_group("Student policy configurations")
        # Value of the following fields should come from the environment or processed.
        non_configurable_fields = {
            "num_policy_obs",
            "num_student_obs",
            "num_actions",
            "teacher_policy",
            "student_policy_path",
        }
        for field_info in fields(StudentPolicyTrainerCfg):
            if field_info.name in non_configurable_fields:
                continue

            # Set the argument name
            arg_name = f"--{StudentPolicyTrainerCfg.args_prefix()}{field_info.name}"
            arg_type = field_info.type
            arg_help = field_info.metadata.get("description", "")

            # Handle default values and types
            default = None
            if field_info.name in default_overwrites:
                default = default_overwrites[field_info.name]
            elif field_info.default is not MISSING:
                default = field_info.default
            elif field_info.default_factory is not MISSING:
                default = field_info.default_factory()

            # Add the argument to the parser
            group.add_argument(arg_name, type=arg_type, default=default, help=arg_help)
