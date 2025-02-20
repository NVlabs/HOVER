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


from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from teacher_policy_cfg import TeacherPolicyCfg

    from neural_wbc.core import EnvironmentWrapper


def get_customized_rsl_rl():
    """Helper function to ensure the correct version of rsl_rl is imported.

    This function does the following:
    1. Gets the installed rsl_rl package location and adds it to sys.path
    2. Removes any existing rsl_rl and submodules from sys.modules to force reimporting
    """
    import sys

    import pkg_resources

    dist = pkg_resources.require("rsl_rl")[0]
    sys.path.insert(0, dist.location)

    # Remove 'rsl_rl' from sys.modules if it was already imported
    modules_to_remove = [key for key in sys.modules if key.startswith("rsl_rl")]
    for module in modules_to_remove:
        print(f"Removing {module} from sys.modules")
        del sys.modules[module]


def get_player_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
    parser.add_argument("--env_spacing", type=int, default=5, help="Distance between environments in simulator.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--reference_motion_path", type=str, default=None, help="Path to the reference motion dataset.")
    parser.add_argument("--robot", type=str, choices=["h1", "gr1"], default="h1", help="Robot used in environment")
    parser.add_argument(
        "--student_player", action="store_true", help="Whether the evaluated policy is a student policy."
    )
    parser.add_argument("--student_path", type=str, default=None, help="The path for the student policy.")
    parser.add_argument("--student_checkpoint", type=str, default=None, help="The exact checkpoint.")
    return parser


get_customized_rsl_rl()
from rsl_rl.runners import OnPolicyRunner


def get_ppo_runner_and_checkpoint_path(
    teacher_policy_cfg: TeacherPolicyCfg,
    wrapped_env: EnvironmentWrapper,
    device: torch.device,
    log_dir: str | None = None,
) -> tuple[OnPolicyRunner, str]:
    resume_path = teacher_policy_cfg.runner.resume_path
    if not resume_path:
        raise ValueError("teacher_policy.resume_path is not specified")
    checkpoint = teacher_policy_cfg.runner.checkpoint
    if not checkpoint:
        raise ValueError("teacher_policy.checkpoint is not specified")
    # specify directory for logging experiments
    print(f"[INFO] Loading experiment from directory: {teacher_policy_cfg.runner.path}")
    checkpoint_path = os.path.join(resume_path, teacher_policy_cfg.runner.checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")

    # Try overwrite policy configuration with the content from {log_root_path}/config.json.
    teacher_policy_cfg.overwrite_policy_cfg_from_file(os.path.join(resume_path, "config.json"))

    # load previously trained model
    ppo_runner = OnPolicyRunner(wrapped_env, teacher_policy_cfg.to_dict(), log_dir=log_dir, device=device)

    return ppo_runner, checkpoint_path
