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


"""Common functions that can be used to enable different controller.

Controller will compute the torque that will be applied to the joint. The implemented controller are PD controllers that track
desire pos/vel and torques. By using different controller, the trained RL policy will be considered outputting different
action. For example, when using Pos Controller, the trained policy is considered outputting desire pos.

"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def position_pd_control(
    env: DirectRLEnv, actions_scaled: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joint_ids=None
):
    asset: Articulation = env.scene.articulations[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel

    if joint_ids:
        default_joint_pos = default_joint_pos[:, joint_ids]
        joint_pos = joint_pos[:, joint_ids]
        joint_vel = joint_vel[:, joint_ids]

    torques = (
        env.kp_scale * env._p_gains * (actions_scaled + default_joint_pos - joint_pos)
        - env.kd_scale * env._d_gains * joint_vel
    )
    return torques


def velocity_pd_control(
    env: DirectRLEnv, actions_scaled: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joint_ids=None
):
    asset: Articulation = env.scene.articulations[asset_cfg.name]
    joint_vel = asset.data.joint_vel
    joint_acc = asset.data.joint_acc

    if joint_ids:
        joint_vel = joint_vel[:, joint_ids]
        joint_acc = joint_acc[:, joint_ids]

    torques = env.kp_scale * env._p_gains * (actions_scaled - joint_vel) - env.kd_scale * env._d_gains * joint_acc

    return torques


def torque_control(
    env: DirectRLEnv, actions_scaled: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joint_ids=None
):
    torques = actions_scaled

    return torques


def resolve_control_fn(
    control_type: Literal["Pos", "Vel", "Torque"] = "Pos",
):
    control_fn = position_pd_control

    if control_type == "Pos":
        print("[INFO]: Setting up pos control")
        control_fn = position_pd_control
    elif control_type == "Vel":
        print("[INFO]: Setting up vel control")
        control_fn = velocity_pd_control
    elif control_type == "Torque":
        print("[INFO]: Setting up torque control")
        control_fn = torque_control
    else:
        raise ValueError(f"Unrecognized control type {control_type}")

    return control_fn
