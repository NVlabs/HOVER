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

import phc.utils.torch_utils as torch_utils

from neural_wbc.core import math_utils

from ..body_state import BodyState
from ..reference_motion import ReferenceMotionState


def compute_teacher_observations(
    body_state: BodyState,
    ref_motion_state: ReferenceMotionState,
    tracked_body_ids: list[int],
    last_actions: torch.Tensor,
    ref_episodic_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes the observations for the teacher model based on the current body state, reference motion state,
    and other relevant parameters.

    Args:
        body_state (BodyState): The current state of the humanoid bodies.
        ref_motion_state (ReferenceMotionState): The reference motion state for the humanoid to track.
        tracked_body_ids (list[int]): List of body IDs to be tracked in observations.
        last_actions (torch.Tensor): The last actions taken.
        ref_episodic_offset (torch.Tensor | None, optional): Episodic offset for the reference motion.
            Defaults to None.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing the concatenated observations tensor and a
            dictionary of individual observations.
    """
    obs_dict = {
        "robot_state": compute_robot_state_observation(body_state),
        "imitation": compute_imitation_observations(
            body_state=body_state,
            ref_motion_state=ref_motion_state,
            tracked_body_ids=tracked_body_ids,
            ref_episodic_offset=ref_episodic_offset,
        ),
        "last_action": last_actions,
    }

    obs = torch.cat(
        [tensor for tensor in obs_dict.values()],
        dim=-1,
    )

    return obs, obs_dict


def compute_robot_state_observation(
    body_state: BodyState,
) -> torch.Tensor:
    """Computes the robot state observation in the robot root frame.

    Args:
        body_state (BodyState): The current state of the humanoid bodies.
    """
    # for normalization
    root_pos = body_state.body_pos_extend[:, 0, :].clone()
    root_rot = body_state.body_rot_extend[:, 0, :].clone()
    body_pos = body_state.body_pos_extend
    body_rot = body_state.body_rot_extend
    body_vel = body_state.body_lin_vel_extend
    body_ang_vel = body_state.body_ang_vel_extend
    num_envs, num_bodies, _ = body_pos.shape

    root_rot = math_utils.convert_quat(root_rot, to="xyzw")
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(num_envs * num_bodies, 4)

    # body pos and normalize to egocentric (for angle only yaw)
    root_pos_extend = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_extend
    flat_local_body_pos = local_body_pos.reshape(num_envs * num_bodies, 3)
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)  # input xyzw
    local_body_pos = flat_local_body_pos.reshape(num_envs, num_bodies * 3)
    local_body_pos_obs = local_body_pos[..., 3:]  # remove root pos

    # body quat and normalize to egocentric (for angle only yaw)
    flat_body_rot = body_rot.reshape(num_envs * num_bodies, 4)
    flat_local_body_rot = math_utils.quat_mul(
        math_utils.convert_quat(flat_heading_rot_inv, to="wxyz"), flat_body_rot
    )  # input wxyz, output wxyz
    flat_local_body_rot = math_utils.convert_quat(flat_local_body_rot, to="xyzw")
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(
        flat_local_body_rot
    )  # Shape becomes (num_envs, num_bodies * 6)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(num_envs, num_bodies * flat_local_body_rot_obs.shape[1])

    # body vel and normalize to egocentric (for angle only yaw)
    flat_body_vel = body_vel.reshape(num_envs * num_bodies, 3)
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(num_envs, num_bodies * 3)

    # body ang vel and normalize to egocentric (for angle only yaw)
    flat_body_ang_vel = body_ang_vel.reshape(num_envs * num_bodies, 3)
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(num_envs, num_bodies * 3)

    return torch.cat([local_body_pos_obs, local_body_rot_obs, local_body_vel, local_body_ang_vel], dim=-1)


def compute_imitation_observations(
    body_state: BodyState,
    ref_motion_state: ReferenceMotionState,
    tracked_body_ids: list[int],
    ref_episodic_offset: torch.Tensor | None = None,
):
    """Computes the imitation observations by comparing the current body state to the reference motion state.

    Args:
        body_state (BodyState): The current state of the humanoid bodies.
        ref_motion_state (ReferenceMotionState): The reference motion state for the humanoid to track.
        tracked_body_ids (list[int]): List of body IDs to be tracked in observations.
        ref_episodic_offset (torch.Tensor | None, optional): Episodic offset for the reference motion.
            Defaults to None.
    """

    # Extract robot state from scene asset
    body_pos = body_state.body_pos_extend[:, tracked_body_ids, :]
    body_rot = body_state.body_rot_extend[:, tracked_body_ids, :]
    body_vel = body_state.body_lin_vel_extend[:, tracked_body_ids, :]
    body_ang_vel = body_state.body_ang_vel_extend[:, tracked_body_ids, :]
    root_pos = body_state.body_pos_extend[:, 0, :].clone()
    root_rot = body_state.body_rot_extend[:, 0, :].clone()

    # Extract reference body state
    ref_body_pos = ref_motion_state.body_pos_extend[:, tracked_body_ids, :]
    ref_body_rot = ref_motion_state.body_rot_extend[:, tracked_body_ids, :]
    ref_body_vel = ref_motion_state.body_lin_vel_extend[:, tracked_body_ids, :]
    ref_body_ang_vel = ref_motion_state.body_ang_vel_extend[:, tracked_body_ids, :]

    num_envs, num_bodies, _ = body_pos.shape

    # torch_utils from IsaacGym using xyzw.
    # IsaacLab and math_utils assume wxyz.
    # we need do a conversion here.
    root_rot = math_utils.convert_quat(root_rot, to="xyzw")

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)  # xyzw
    heading_rot = torch_utils.calc_heading_quat(root_rot)  # xyzw
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, num_bodies, 1))
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, num_bodies, 1))

    # Body position differences
    diff_global_body_pos = ref_body_pos - body_pos
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(
        heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3)
    )  # input xyzw

    # Body rotation differences
    diff_global_body_rot = math_utils.quat_mul(
        ref_body_rot,
        math_utils.quat_conjugate(body_rot),
    )  # input wxyz out wxyz
    diff_local_body_rot_flat = math_utils.quat_mul(
        math_utils.quat_mul(
            math_utils.convert_quat(heading_inv_rot_expand.view(-1, 4), to="wxyz"), diff_global_body_rot.view(-1, 4)
        ),
        math_utils.convert_quat(heading_rot_expand.view(-1, 4), to="wxyz"),
    )  # Need to be change of basis  # input wxyz
    diff_local_body_rot_flat = math_utils.convert_quat(diff_local_body_rot_flat, to="xyzw")  # out xyzw

    # linear Velocity differences
    diff_global_vel = ref_body_vel - body_vel
    diff_local_vel = torch_utils.my_quat_rotate(
        heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3)
    )  # input xyzw

    # Angular Velocity differences
    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_local_ang_vel = torch_utils.my_quat_rotate(
        heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3)
    )  # input xyzw

    local_ref_body_pos = ref_body_pos - root_pos.view(num_envs, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(
        heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3)
    )  # input xyzw

    local_ref_body_rot = math_utils.quat_mul(
        math_utils.convert_quat(heading_inv_rot_expand.view(-1, 4), to="wxyz"), ref_body_rot.view(-1, 4)
    )  # input wxyz
    local_ref_body_rot = math_utils.convert_quat(local_ref_body_rot, to="xyzw")  # out xyzw
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)  # out xyzw

    if ref_episodic_offset is not None:
        diff_global_body_pos_offset = ref_episodic_offset.unsqueeze(1).unsqueeze(2).expand(-1, 1, num_bodies, -1)
        diff_local_body_pos_flat = (
            diff_local_body_pos_flat.view(num_envs, 1, num_bodies, 3) + diff_global_body_pos_offset
        )
        local_ref_body_pos_offset = ref_episodic_offset.repeat(num_bodies, 1)[
            : num_bodies * ref_episodic_offset.shape[0], :
        ]
        local_ref_body_pos += local_ref_body_pos_offset

    obs = [
        diff_local_body_pos_flat.view(num_envs, -1),  # num_envs * num_bodies * 3
        torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(num_envs, -1),  # num_envs * num_bodies * 6
        diff_local_vel.view(num_envs, -1),  # num_envs * num_bodies * 3
        diff_local_ang_vel.view(num_envs, -1),  # num_envs * num_bodies * 3
        local_ref_body_pos.view(num_envs, -1),  # num_envs * num_bodies * 3
        local_ref_body_rot.view(num_envs, -1),  # num_envs * num_bodies * 6
    ]

    obs = torch.cat(obs, dim=-1).view(num_envs, -1)
    return obs
