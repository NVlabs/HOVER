# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from inference_env.deployment_player import DeploymentPlayer
from inference_env.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1
from inference_env.utils import get_player_args

from neural_wbc.core.evaluator import Evaluator
from neural_wbc.data import get_data_path

# add argparse arguments
parser = get_player_args(description="Evaluates motion tracking policy and collects metrics in MuJoCo.")
parser.add_argument("--metrics_path", type=str, default=None, help="Path to store metrics in.")
args_cli = parser.parse_args()


def main():
    custom_config = None
    if args_cli.env_config_overwrite is not None:
        with open(args_cli.env_config_overwrite) as fh:
            custom_config = yaml.safe_load(fh)
        print("[INFO]: Using custom configuration:")
        pprint.pprint(custom_config)

    # Initialize the player with the updated properties
    player = DeploymentPlayer(
        args_cli=args_cli,
        env_cfg=NeuralWBCEnvCfgH1(model_xml_path=get_data_path("mujoco/models/scene.xml")),
        custom_config=custom_config,
    )
    evaluator = Evaluator(env_wrapper=player.env, metrics_path=args_cli.metrics_path)

    should_stop = False
    while not should_stop:
        _, obs, dones, extras = player.play_once()

        reset_env = evaluator.collect(dones=dones, info=extras)
        if reset_env and not evaluator.is_evaluation_complete():
            evaluator.forward_motion_samples()
            _ = player.reset()

        should_stop = evaluator.is_evaluation_complete()

    evaluator.conclude()


if __name__ == "__main__":
    main()
