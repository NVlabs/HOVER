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
import time
import torch
import yaml

from inference_env.deployment_player import DeploymentPlayer
from inference_env.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1
from inference_env.utils import get_player_args

from neural_wbc.data import get_data_path

# add argparse arguments
parser = get_player_args(description="Evaluates motion tracking policy and collects metrics in MuJoCo.")
args_cli = parser.parse_args()

CTRL_FREQ = 200.0


def prep_joint_pos_cmd(env, joint_pos_cmd: dict[str, float] | None = None) -> torch.Tensor:
    full_joint_pos_cmd = env.robot.joint_positions
    if joint_pos_cmd is not None:
        with torch.inference_mode():
            for joint_name, joint_cmd in joint_pos_cmd.items():
                joint_id = env.robot.get_joint_ids(joint_names=[joint_name])[joint_name]
                full_joint_pos_cmd[:, joint_id] = joint_cmd
    return full_joint_pos_cmd


def main():
    custom_config = None
    if args_cli.env_config_overwrite is not None:
        with open(args_cli.env_config_overwrite) as fh:
            custom_config = yaml.safe_load(fh)
        print("[INFO]: Using custom configuration:")
        pprint.pprint(custom_config)

    # Delta joint position command of the robot. These will be applied based on the PD positional controller
    # defined in control.py.
    joint_pos_cmd = {
        "right_elbow": 2.0,
        "left_elbow": -2.0,
        "torso": 2.0,
        "right_knee": 2.0,
        "left_knee": 2.0,
        "right_ankle": 1.0,
        "left_ankle": 1.0,
    }

    player = DeploymentPlayer(
        args_cli=args_cli,
        env_cfg=NeuralWBCEnvCfgH1(model_xml_path=get_data_path("mujoco/models/scene.xml")),
        custom_config=custom_config,
        demo_mode=True,
    )
    for i in range(args_cli.max_iterations):
        _, obs, dones, extras = player.play_once(prep_joint_pos_cmd(player.env, joint_pos_cmd))
        time.sleep(1.0 / CTRL_FREQ)


if __name__ == "__main__":
    # Run the main execution
    main()
