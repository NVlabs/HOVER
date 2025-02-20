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


import torch
from dataclasses import dataclass

from inference_env.neural_wbc_env_cfg import NeuralWBCEnvCfg

from neural_wbc.core.mask import calculate_mask_length
from neural_wbc.data import get_data_path


@dataclass
class NeuralWBCEnvCfgH1(NeuralWBCEnvCfg):
    decimation = 4
    dt = 0.005
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [2, 2]
    default_rfi_lim = 0
    robot = "mujoco_robot"

    extend_body_parent_names = ["left_elbow_link", "right_elbow_link", "pelvis"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]])

    tracked_body_names = [
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

    # Distillation parameters:
    single_history_dim = 63
    observation_history_length = 25
    num_bodies = 20
    num_joints = 19
    mask_length = calculate_mask_length(
        num_bodies=num_bodies + len(extend_body_parent_names),
        num_joints=num_joints,
    )

    control_type = "Pos"
    robot_actuation_type = "Torque"  # Pos or Torque

    # control parameters
    stiffness = {
        "left_hip_yaw": 150.0,
        "left_hip_roll": 150.0,
        "left_hip_pitch": 200.0,
        "left_knee": 200.0,
        "left_ankle": 20.0,
        "right_hip_yaw": 150.0,
        "right_hip_roll": 150.0,
        "right_hip_pitch": 200.0,
        "right_knee": 200.0,
        "right_ankle": 20.0,
        "torso": 200.0,
        "left_shoulder_pitch": 40.0,
        "left_shoulder_roll": 40.0,
        "left_shoulder_yaw": 40.0,
        "left_elbow": 40.0,
        "right_shoulder_pitch": 40.0,
        "right_shoulder_roll": 40.0,
        "right_shoulder_yaw": 40.0,
        "right_elbow": 40.0,
    }

    damping = {
        "left_hip_yaw": 5.0,
        "left_hip_roll": 5.0,
        "left_hip_pitch": 5.0,
        "left_knee": 5.0,
        "left_ankle": 4.0,
        "right_hip_yaw": 5.0,
        "right_hip_roll": 5.0,
        "right_hip_pitch": 5.0,
        "right_knee": 5.0,
        "right_ankle": 4.0,
        "torso": 5.0,
        "left_shoulder_pitch": 10.0,
        "left_shoulder_roll": 10.0,
        "left_shoulder_yaw": 10.0,
        "left_elbow": 10.0,
        "right_shoulder_pitch": 10.0,
        "right_shoulder_roll": 10.0,
        "right_shoulder_yaw": 10.0,
        "right_elbow": 10.0,
    }

    effort_limit = {
        "left_hip_yaw": 200.0,
        "left_hip_roll": 200.0,
        "left_hip_pitch": 200.0,
        "left_knee": 300.0,
        "left_ankle": 40.0,
        "right_hip_yaw": 200.0,
        "right_hip_roll": 200.0,
        "right_hip_pitch": 200.0,
        "right_knee": 300.0,
        "right_ankle": 40.0,
        "torso": 200.0,
        "left_shoulder_pitch": 40.0,
        "left_shoulder_roll": 40.0,
        "left_shoulder_yaw": 18.0,
        "left_elbow": 18.0,
        "right_shoulder_pitch": 40.0,
        "right_shoulder_roll": 40.0,
        "right_shoulder_yaw": 18.0,
        "right_elbow": 18.0,
    }

    position_limit = {
        "left_hip_yaw": [-0.43, 0.43],
        "left_hip_roll": [-0.43, 0.43],
        "left_hip_pitch": [-1.57, 1.57],
        "left_knee": [-0.26, 2.05],
        "left_ankle": [-0.87, 0.52],
        "right_hip_yaw": [-0.43, 0.43],
        "right_hip_roll": [-0.43, 0.43],
        "right_hip_pitch": [-1.57, 1.57],
        "right_knee": [-0.26, 2.05],
        "right_ankle": [-0.87, 0.52],
        "torso": [-2.35, 2.35],
        "left_shoulder_pitch": [-2.87, 2.87],
        "left_shoulder_roll": [-0.34, 3.11],
        "left_shoulder_yaw": [-1.3, 4.45],
        "left_elbow": [-1.25, 2.61],
        "right_shoulder_pitch": [-2.87, 2.87],
        "right_shoulder_roll": [-3.11, 0.34],
        "right_shoulder_yaw": [-4.45, 1.3],
        "right_elbow": [-1.25, 2.61],
    }

    robot_init_state = {
        "base_pos": [0.0, 0.0, 1.05],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "joint_pos": {
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": -0.28,
            "left_knee": 0.79,
            "left_ankle": -0.52,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -0.28,
            "right_knee": 0.79,
            "right_ankle": -0.52,
            "torso": 0.0,
            "left_shoulder_pitch": 0.28,
            "left_shoulder_roll": 0.0,
            "left_shoulder_yaw": 0.0,
            "left_elbow": 0.52,
            "right_shoulder_pitch": 0.28,
            "right_shoulder_roll": 0.0,
            "right_shoulder_yaw": 0.0,
            "right_elbow": 0.52,
        },
        "joint_vel": {},
    }

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # hips, knees, ankles
    upper_body_joint_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18]  # torso, shoulders, elbows

    def __post_init__(self):
        self.reference_motion_cfg.motion_path = get_data_path("motions/stable_punch.pkl")
        self.reference_motion_cfg.skeleton_path = get_data_path("motion_lib/h1.xml")
