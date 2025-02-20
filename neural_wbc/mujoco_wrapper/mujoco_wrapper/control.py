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


from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mujoco_wrapper.neural_wbc_env import NeuralWBCEnv


def position_pd_control(env: NeuralWBCEnv, pos_actions: torch.Tensor, joint_ids=None):
    """Calculates the PD control torque based on the network output position actions"""
    robot = env.robot
    joint_pos = robot.joint_positions
    joint_vel = robot.joint_velocities

    if joint_ids:
        joint_pos = joint_pos[:, joint_ids]
        joint_vel = joint_vel[:, joint_ids]

    torques = env.p_gains * (pos_actions - joint_pos) - env.d_gains * joint_vel
    return torques


def torque_control(env: NeuralWBCEnv, torque_actions: torch.Tensor, joint_ids=None):
    """Calculates desired torques based on the output of the network"""
    torques = torque_actions

    return torques


def null_control(env: NeuralWBCEnv, actions_scaled: torch.Tensor, joint_ids=None):
    return actions_scaled


def resolve_control_fn(
    control_type: Literal["Pos", "Vel", "Torque", "None"] = "Pos",
):
    control_fn = position_pd_control

    if control_type == "Pos":
        print("[INFO]: Setting up pos control")
        control_fn = position_pd_control
    elif control_type == "Torque":
        print("[INFO]: Setting up torque control")
        control_fn = torque_control
    elif control_type == "None":
        print("[INFO]: Setting up no control")
        control_fn = null_control
    else:
        raise ValueError(f"Unrecognized control type {control_type}")

    return control_fn
