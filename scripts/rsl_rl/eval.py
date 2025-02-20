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


import pprint
import yaml

from teacher_policy_cfg import TeacherPolicyCfg

from isaaclab.app import AppLauncher

# local imports
from utils import get_player_args  # isort: skip

# add argparse arguments
parser = get_player_args(description="Evaluates motion tracking policy and collects metrics.")
parser.add_argument("--metrics_path", type=str, default=None, help="Path to store metrics in.")
parser.add_argument(
    "--env_config_overwrite", type=str, default=None, help="Path to yaml file with overwriting configuration values."
)

# append RSL-RL cli arguments
TeacherPolicyCfg.add_args_to_parser(parser=parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from players import EvaluationPlayer


def main():
    custom_config = None
    if args_cli.env_config_overwrite is not None:
        with open(args_cli.env_config_overwrite) as fh:
            custom_config = yaml.safe_load(fh)
        print("[INFO]: Using custom configuration:")
        pprint.pprint(custom_config)

    # Initialize the player with the updated properties
    player = EvaluationPlayer(args_cli=args_cli, metrics_path=args_cli.metrics_path, custom_config=custom_config)
    player.play(simulation_app=simulation_app)


if __name__ == "__main__":
    # Run the main execution
    main()
    # Close sim app
    simulation_app.close()
