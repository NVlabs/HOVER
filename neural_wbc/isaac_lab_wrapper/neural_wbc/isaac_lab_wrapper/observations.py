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
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from .neural_wbc_env import NeuralWBCEnv

from neural_wbc.core import ReferenceMotionState
from neural_wbc.core.body_state import BodyState
from neural_wbc.core.observations import compute_student_observations, compute_teacher_observations


def compute_observations(
    env: NeuralWBCEnv,
    body_state: BodyState,
    ref_motion_state: ReferenceMotionState,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene.articulations[asset_cfg.name]

    obs_dict = {}

    # First collect teacher policy observations.
    teacher_obs, teacher_obs_dict = compute_teacher_observations(
        body_state=body_state,
        ref_motion_state=ref_motion_state,
        tracked_body_ids=env._tracked_body_ids,
        ref_episodic_offset=env.ref_episodic_offset,
        last_actions=env.actions,
    )
    obs_dict.update(teacher_obs_dict)
    obs_dict["teacher_policy"] = teacher_obs

    # Then the privileged observations.
    privileged_obs, privileged_obs_dict = compute_privileged_observations(env=env, asset=asset)
    obs_dict.update(privileged_obs_dict)
    obs_dict["critic"] = torch.cat([teacher_obs, privileged_obs], dim=1)

    # If we are in a distill mode, add student observations.
    if env.cfg.mode.is_distill_mode():
        base_id = env._body_names.index(env.base_name)
        student_obs, student_obs_dict = compute_student_observations(
            base_id=base_id,
            body_state=body_state,
            ref_motion_state=ref_motion_state,
            projected_gravity=asset.data.projected_gravity_b,
            last_actions=env.actions,
            history=env.history.entries,
            mask=env.mask,
            ref_episodic_offset=env.ref_episodic_offset,
        )

        obs_dict.update(student_obs_dict)
        obs_dict["student_policy"] = student_obs

    return obs_dict


def compute_privileged_observations(env: NeuralWBCEnv, asset: Articulation):
    contact_forces = env.contact_sensor.data.net_forces_w[:, env.feet_ids, :]

    privileged_obs_dict = {
        "base_com_bias": env.base_com_bias.to(env.device),
        "ground_friction_values": asset.data.joint_friction[:, env.feet_ids],
        "body_mass_scale": env.body_mass_scale,
        "kp_scale": env.kp_scale,
        "kd_scale": env.kd_scale,
        "rfi_lim_scale": env.rfi_lim / env.cfg.default_rfi_lim,
        "contact_forces": contact_forces.reshape(contact_forces.shape[0], -1),
        "recovery_counters": torch.clamp_max(env.recovery_counters.unsqueeze(1), 1),
    }
    privileged_obs = torch.cat(
        [tensor for tensor in privileged_obs_dict.values()],
        dim=-1,
    )
    return privileged_obs, privileged_obs_dict
