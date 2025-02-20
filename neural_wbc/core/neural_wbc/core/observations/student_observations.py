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

import phc.utils.torch_utils as torch_utils

from neural_wbc.core import math_utils

from ..body_state import BodyState
from ..reference_motion import ReferenceMotionState


def compute_student_observations(
    base_id: int,
    body_state: BodyState,
    ref_motion_state: ReferenceMotionState,
    projected_gravity: torch.Tensor,
    last_actions: torch.Tensor,
    history: torch.Tensor,
    mask: torch.Tensor,
    ref_episodic_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes observations for a student policy."""
    obs_dict = {
        "distilled_robot_state": compute_distilled_robot_state_observation(
            body_state=body_state,
            base_id=base_id,
            projected_gravity=projected_gravity,
        ),
        "distilled_imitation": compute_distilled_imitation_observations(
            ref_motion_state=ref_motion_state,
            body_state=body_state,
            mask=mask,
            ref_episodic_offset=ref_episodic_offset,
        ),
        "distilled_last_action": last_actions,
        "distilled_historical_info": history,
    }

    obs = torch.cat(
        [tensor for tensor in obs_dict.values()],
        dim=-1,
    )

    return obs, obs_dict


def compute_distilled_imitation_observations(
    ref_motion_state: ReferenceMotionState,
    body_state: BodyState,
    mask: torch.Tensor,
    ref_episodic_offset: torch.Tensor | None,
) -> torch.Tensor:
    """Computes the reference goal state used in the observation of the student."""
    # First we get all reference states.
    kinematic_command = compute_kinematic_command(ref_motion_state, body_state, ref_episodic_offset)
    joint_command = compute_joint_command(ref_motion_state, body_state)
    root_command = compute_root_command(ref_motion_state, body_state)

    # Apply masking to kinematic references. The mask contains 1 value for every kinematic body, but
    # in the observation every kinematic body corresponds to 3 values (x,y,z). For this we repeat
    # the kinematic part of the mask 3 times.
    num_bodies = kinematic_command.shape[1] // 3
    kinematic_mask = mask[:, :num_bodies].repeat_interleave(3, dim=-1)
    kinematic_command *= kinematic_mask

    # Apply masking to joint references.
    num_joints = joint_command.shape[1]
    joint_mask = mask[:, num_bodies : num_bodies + num_joints]
    joint_command *= joint_mask

    # Apply masking to root references.
    root_mask = mask[:, num_bodies + num_joints :]
    root_command *= root_mask

    # Concatenate all references. We also have to add the mask itself, as else the network has no
    # way to determine the difference between a target state that is enabled but set to 0, or a
    # target state that is 0 because it is disabled.
    observations = torch.cat(
        [
            kinematic_command,
            joint_command,
            root_command,
            mask,
        ],
        dim=1,
    )

    return observations


def compute_kinematic_command(
    ref_motion_state: ReferenceMotionState,
    body_state: BodyState,
    ref_episodic_offset: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute the link position command used in the observation of the student.

    The link position command consists of:
    - the delta between the current root position and the target link positions
    """
    num_envs, num_bodies, _ = body_state.body_pos_extend.shape

    root_pos = body_state.body_pos_extend[:, 0, :]
    root_rot_wxyz = body_state.body_rot_extend[:, 0, :]
    root_rot_xyzw = math_utils.convert_quat(root_rot_wxyz, to="xyzw")
    ref_body_pos = ref_motion_state.body_pos_extend

    heading_inv_rot_xyzw = torch_utils.calc_heading_quat_inv(root_rot_xyzw)
    heading_inv_rot_expand_xyzw = heading_inv_rot_xyzw.unsqueeze(-2).repeat((1, num_bodies, 1))

    # Delta between the current root position and the target link positions.
    local_ref_body_pos = ref_body_pos - root_pos.unsqueeze(1).expand(-1, num_bodies, -1)
    local_ref_body_pos = torch_utils.my_quat_rotate(
        heading_inv_rot_expand_xyzw.view(-1, 4),
        local_ref_body_pos.view(-1, 3),
    ).view(num_envs, num_bodies, -1)

    return torch.cat(
        [
            local_ref_body_pos.view(num_envs, -1),  # num_envs * (num_bodies * 3)
        ],
        dim=1,
    )


def compute_joint_command(ref_motion_state: ReferenceMotionState, body_state: BodyState) -> torch.Tensor:
    """
    Compute the joint command used in the observation of the student.

    The joint reference is the delta between the current joint position/velocity and the target
    joint position/velocity.
    """
    delta_joint_pos = ref_motion_state.joint_pos - body_state.joint_pos
    return torch.cat([delta_joint_pos], dim=-1)


def compute_root_command(ref_motion_state: ReferenceMotionState, body_state: BodyState) -> torch.Tensor:
    """
    Compute the root command used in the observation of the student.

    The root command consists of
    - the target root velocity (in the root frame)
    - the target root roll and pitch
    - the delta between the current root yaw and the target root yaw
    - the root height.
    """
    target_root_linear_velocity = math_utils.quat_rotate_inverse(
        ref_motion_state.root_rot, ref_motion_state.root_lin_vel
    )

    ref_root_rot_wxyz = ref_motion_state.root_rot
    ref_root_rot_rpy = math_utils.euler_xyz_from_quat(ref_root_rot_wxyz)

    root_rot_wxyz = body_state.body_rot[:, 0, :]
    root_rot_rpy = math_utils.euler_xyz_from_quat(root_rot_wxyz)

    # We use the yaw_delta since yaw cannot be estimated from just proprioceptive measurements.
    # Another alternative option could be to use yaw velocity instead.
    roll = ref_root_rot_rpy[0]
    pitch = ref_root_rot_rpy[1]
    delta_yaw = ref_root_rot_rpy[2] - root_rot_rpy[2]
    target_root_rot_rpy = torch.stack([roll, pitch, delta_yaw], dim=-1)

    target_root_height = ref_motion_state.body_pos[:, 0, 2]

    return torch.cat(
        [
            target_root_linear_velocity,  # num_envs * 3
            target_root_rot_rpy,  # num_envs * 3
            target_root_height.view(-1, 1),  # num_envs * 1
        ],
        dim=1,
    )


def compute_distilled_robot_state_observation(
    body_state: BodyState,
    base_id: int,
    projected_gravity: torch.Tensor,
) -> torch.Tensor:
    """Root body state in the robot root frame."""
    # for normalization
    joint_pos = body_state.joint_pos.clone()
    joint_vel = body_state.joint_vel.clone()

    base_rot = body_state.body_rot_extend[:, base_id, :].clone()
    base_ang_vel = body_state.body_ang_vel_extend[:, base_id, :].clone()
    local_base_ang_vel = math_utils.quat_rotate_inverse(base_rot, base_ang_vel)

    return torch.cat([joint_pos, joint_vel, local_base_ang_vel, projected_gravity], dim=-1)
