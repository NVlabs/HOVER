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

from isaaclab.app import AppLauncher

# local imports
from utils import get_player_args  # isort: skip
import cli_args  # isort: skip


# add argparse arguments
parser = get_player_args(description="Plays motion tracking policy in Isaac Lab.")
parser.add_argument("--randomize", action="store_true", help="Whether to randomize reference motion while playing.")
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from players import DemoPlayer


def main():
    player = DemoPlayer(args_cli=args_cli, randomize=args_cli.randomize)
    player.play(simulation_app=simulation_app)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
