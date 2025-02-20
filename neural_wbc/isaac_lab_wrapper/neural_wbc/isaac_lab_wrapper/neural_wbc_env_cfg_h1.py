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

import torch

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.data import get_data_path

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_assets import H1_CFG

from .events import NeuralWBCPlayEventCfg, NeuralWBCTrainEventCfg
from .neural_wbc_env_cfg import NeuralWBCEnvCfg
from .terrain import HARD_ROUGH_TERRAINS_CFG, flat_terrain

DISTILL_MASK_MODES_ALL = {
    "exbody": {
        "upper_body": [".*torso_joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": ["root.*"],
    },
    "humanplus": {
        "upper_body": [".*torso_joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": [".*hip.*joint.*", ".*knee.*joint.*", ".*ankle.*joint.*", "root.*"],
    },
    "h2o": {
        "upper_body": [
            ".*shoulder.*link.*",
            ".*elbow.*link.*",
            ".*hand.*link.*",
        ],
        "lower_body": [".*ankle.*link.*"],
    },
    "omnih2o": {
        "upper_body": [".*hand.*link.*", ".*head.*link.*"],
    },
}


@configclass
class NeuralWBCEnvCfgH1(NeuralWBCEnvCfg):
    # General parameters:
    action_space = 19
    observation_space = 913
    state_space = 990

    # Distillation parameters:
    single_history_dim = 63
    observation_history_length = 25

    # Mask setup for an OH2O specialist policy as default:
    # OH2O mode is tracking the head and hand positions. This can be modified to train a different specialist
    # or use the full DISTILL_MASK_MODES_ALL to train a generalist policy.
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}

    # Robot geometry / actuation parameters:
    actuators = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
            effort_limit={
                ".*_hip_yaw": 200.0,
                ".*_hip_roll": 200.0,
                ".*_hip_pitch": 200.0,
                ".*_knee": 300.0,
                "torso": 200.0,
            },
            velocity_limit={
                ".*_hip_yaw": 23.0,
                ".*_hip_roll": 23.0,
                ".*_hip_pitch": 23.0,
                ".*_knee": 14.0,
                "torso": 23.0,
            },
            stiffness=0,
            damping=0,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit=40,
            velocity_limit=9.0,
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit={
                ".*_shoulder_pitch": 40.0,
                ".*_shoulder_roll": 40.0,
                ".*_shoulder_yaw": 18.0,
                ".*_elbow": 18.0,
            },
            velocity_limit={
                ".*_shoulder_pitch": 9.0,
                ".*_shoulder_roll": 9.0,
                ".*_shoulder_yaw": 20.0,
                ".*_elbow": 20.0,
            },
            stiffness=0,
            damping=0,
        ),
    }

    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)

    body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ]

    # Joint names by the order in the MJCF model.
    joint_names = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
        "torso",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
    ]

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # hips, knees, ankles
    upper_body_joint_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18]  # torso, shoulders, elbows

    base_name = "torso_link"

    feet_name = ".*_ankle_link"

    extend_body_parent_names = ["left_elbow_link", "right_elbow_link", "pelvis"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]])

    # These are the bodies that are tracked by the teacher. They may also contain the extended
    # bodies.
    tracked_body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

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

    mass_randomized_body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "torso_link",
    ]

    undesired_contact_body_names = [
        "pelvis",
        ".*_yaw_link",
        ".*_roll_link",
        ".*_pitch_link",
        ".*_knee_link",
    ]

    # Add a height scanner to the torso to detect the height of the terrain mesh
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        # Apply a grid pattern that is smaller than the resolution to only return one height value.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    def __post_init__(self):
        super().__post_init__()

        self.reference_motion_manager.motion_path = get_data_path("motions/stable_punch.pkl")
        self.reference_motion_manager.skeleton_path = get_data_path("motion_lib/h1.xml")

        if self.terrain.terrain_generator == HARD_ROUGH_TERRAINS_CFG:
            self.events.update_curriculum.params["penalty_level_up_threshold"] = 125

        if self.mode == NeuralWBCModes.TRAIN:
            self.episode_length_s = 20.0
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "torso_link"
        elif self.mode == NeuralWBCModes.DISTILL:
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "torso_link"
            self.add_policy_obs_noise = False
            self.reset_mask = True
            # Do not reset mask when there is only one mode.
            num_regions = len(self.distill_mask_modes)
            if num_regions == 1:
                region_modes = list(self.distill_mask_modes.values())[0]
                if len(region_modes) == 1:
                    self.reset_mask = False
        elif self.mode == NeuralWBCModes.TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        elif self.mode == NeuralWBCModes.DISTILL_TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.distill_teleop_selected_keypoints_names = []
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.default_rfi_lim = 0.0
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
