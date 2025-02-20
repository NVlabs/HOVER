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


from dataclasses import dataclass
from typing import Literal, Protocol

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.core.reference_motion import ReferenceMotionManagerCfg


@dataclass
class NeuralWBCEnvCfg(Protocol):
    model_xml_path: str
    enable_viewer: bool = False
    mode = NeuralWBCModes.DISTILL_TEST

    decimation = 4
    dt = 0.005
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [0, 0]
    default_rfi_lim = 0.0

    # Distillation parameters:
    single_history_dim: int = 63
    observation_history_length: int = 25

    reference_motion_cfg = ReferenceMotionManagerCfg()

    # Termination conditions
    gravity_x_threshold = 0.7
    gravity_y_threshold = 0.7
    max_ref_motion_dist = 0.5

    # control type: the action type from the policy
    # "Pos": target joint pos, "Torque": joint torques
    # 'None': by passes control and returns same value
    control_type: Literal["Pos", "Torque", "None"] = "Pos"
