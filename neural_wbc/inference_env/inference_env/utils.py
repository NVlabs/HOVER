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


def get_player_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--headless", action="store_true", help="Whether to show mujoco visualizer.")
    parser.add_argument("--reference_motion_path", type=str, default=None, help="Path to the reference motion dataset.")
    parser.add_argument("--student_path", type=str, default=None, help="The path for the student policy.")
    parser.add_argument("--student_checkpoint", type=str, default=None, help="The exact checkpoint.")
    parser.add_argument("--robot", type=str, default="mujoco_robot", help="The name of registered robot class.")
    parser.add_argument("--max_iterations", type=int, default=100, help="Number steps to run the demo example.")
    parser.add_argument(
        "--env_config_overwrite",
        type=str,
        default=None,
        help="Path to yaml file with overwriting configuration values.",
    )
    return parser
