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

import json
from dataclasses import MISSING

from neural_wbc.core import ReferenceMotionManagerCfg
from neural_wbc.core.modes import NeuralWBCModes

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .events import NeuralWBCEventCfg, NeuralWBCPlayEventCfg, NeuralWBCTrainEventCfg
from .rewards import NeuralWBCRewardCfg
from .terrain import rough_terrain
from .utils import convert_serializable_to_tensors_and_slices, convert_tensors_and_slices_to_serializable


@configclass
class NeuralWBCEnvCfg(DirectRLEnvCfg):
    mode = NeuralWBCModes.TRAIN

    # Environment setup
    episode_length_s = 3600.0
    decimation = 4
    action_scale = 0.25
    dt = 0.005

    # Distillation parameters:
    single_history_dim: int = MISSING
    observation_history_length: int = MISSING
    distill_mask_modes: dict[str, dict[str, list[str]]] = MISSING
    distill_mask_sparsity_randomization_enabled: bool = MISSING

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation, physx=PhysxCfg(bounce_threshold_velocity=0.2))

    # terrain
    terrain = rough_terrain

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=True)

    # Robot
    robot: ArticulationCfg = MISSING

    base_name: str = MISSING
    feet_name: str | list[str] = MISSING
    body_names: list[str] = MISSING
    joint_names: list[str] = MISSING
    tracked_body_names: list[str] = MISSING
    stiffness: dict[str, float] = MISSING
    damping: dict[str, float] = MISSING
    mass_randomized_body_names: list[str] = MISSING
    undesired_contact_body_names: list[str] = MISSING
    extend_body_parent_names: list[str] = []
    extend_body_names: list[str] = []
    extend_body_pos: list[list[float]] = []

    # control type: the action type from the policy
    # "Pos": target joint pos, "Vel": target joint vel, "Torque": joint torques
    control_type = "Pos"

    # Control delay step range (min, max): the control will be randomly delayed at least "min" steps and at most
    # "max" steps. If (0,0), then it means no delay happen
    ctrl_delay_step_range = (0, 3)

    # The default control noise limits: we will add noise to the final torques. the default_rfi_lim defines
    # the default limit of the range of the added noise. It represented by the percentage of the control limits.
    # noise = uniform(-rfi_lim * torque_limits, rfi_lim * torque_limits)
    default_rfi_lim = 0.1

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # Add a height scanner to the torso to detect the height of the terrain mesh
    height_scanner: RayCasterCfg = MISSING

    # domain randomization config
    events: NeuralWBCEventCfg = MISSING

    # Recovery Counter for Pushed robot: Give steps for the robot to stabilize
    recovery_count = 60

    # Termination conditions
    gravity_x_threshold = 0.7
    gravity_y_threshold = 0.7
    max_ref_motion_dist = 1.5

    # reward scales
    rewards: NeuralWBCRewardCfg = NeuralWBCRewardCfg()

    # motion and skeleton files
    reference_motion_manager = ReferenceMotionManagerCfg()

    # When we resample reference motions
    resample_motions = True  # if we want to resample reference motions
    resample_motions_for_envs_interval_s = 1000  # How many seconds between we resample the reference motions

    # observation noise
    add_policy_obs_noise = True
    policy_obs_noise_level = 1.0
    policy_obs_noise_scales = {
        "body_pos": 0.01,  # body pos in cartesian space: 19x3
        "body_rot": 0.01,  # body pos in cartesian space: 19x3
        "body_lin_vel": 0.01,  # body velocity in cartesian space: 19x3
        "body_ang_vel": 0.01,  # body velocity in cartesian space: 19x3
        "ref_body_pos_diff": 0.05,
        "ref_body_rot_diff": 0.01,
        "ref_body_pos": 0.01,
        "ref_body_rot": 0.01,
        "ref_lin_vel": 0.01,
        "ref_ang_vel": 0.01,
    }

    # Whether to reset mask. Default to false as teacher training does not require mask resetting
    reset_mask = False

    def __post_init__(self):
        super().__post_init__()

        if self.mode.is_training_mode():
            self.events = NeuralWBCTrainEventCfg()
        elif self.mode.is_test_mode():
            self.events = NeuralWBCPlayEventCfg()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 10.0
            self.add_policy_obs_noise = False
            self.resample_motions = False
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

    def save(self, file_path: str):
        data = self.to_dict()
        serializable_data = convert_tensors_and_slices_to_serializable(data)

        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(serializable_data, fh)

    def load(self, file_path: str):
        with open(file_path, encoding="utf-8") as fh:
            loaded_dict = json.load(fh)
        config_dict = convert_serializable_to_tensors_and_slices(loaded_dict)
        return self.from_dict(config_dict)
