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

from .body_state import BodyState
from .reference_motion import ReferenceMotionManager, ReferenceMotionState


def check_termination_conditions(
    training_mode: bool,
    projected_gravity: torch.Tensor,
    gravity_x_threshold: float,
    gravity_y_threshold: float,
    ref_motion_mgr: ReferenceMotionManager,
    episode_times: torch.Tensor,
    body_state: BodyState,
    ref_motion_state: ReferenceMotionState,
    max_ref_motion_dist: float,
    in_recovery: torch.Tensor | None,
    mask: torch.Tensor,
    net_contact_forces: torch.Tensor,
    undesired_contact_body_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Evaluates termination conditions.

    This function checks various termination conditions and returns a boolean tensor of shape (num_env,)
    indicating whether any condition has been met. Additionally, it provides a dictionary mapping
    each condition's name to its activation state, with each state represented as a boolean tensor
    of shape (num_env,).

    Returns:
        torch.Tensor: A boolean tensor of shape (num_env,) where a value of True indicates that at least
        one termination condition is active, and False indicates none are active.
        dict: A dictionary where keys are the names of the termination conditions and values are
        boolean tensors of shape (num_env,) indicating whether each condition is active.
    """
    conditions = {
        "gravity": terminate_by_gravity(
            projected_gravity=projected_gravity,
            gravity_x_threshold=gravity_x_threshold,
            gravity_y_threshold=gravity_y_threshold,
        ),
        "undesired_contact": terminate_by_undesired_contact(
            net_contact_forces=net_contact_forces,
            undesired_contact_body_ids=undesired_contact_body_ids,
        ),
        "reference_motion_length": terminate_by_reference_motion_length(
            ref_motion_mgr=ref_motion_mgr, episode_times=episode_times
        ),
        "reference_motion_distance": terminate_by_reference_motion_distance(
            training_mode=training_mode,
            body_state=body_state,
            ref_motion_state=ref_motion_state,
            max_ref_motion_dist=max_ref_motion_dist,
            in_recovery=in_recovery,
            mask=mask,
        ),
    }
    should_terminate = torch.any(torch.cat(list(conditions.values()), dim=1), dim=1)
    return should_terminate, conditions


def terminate_by_gravity(
    projected_gravity: torch.Tensor,
    gravity_x_threshold: float,
    gravity_y_threshold: float,
) -> torch.Tensor:
    """
    Checks termination condition based on robot balance.

    This function evaluates whether the robot is unbalanced due to gravity and returns
    a boolean tensor of shape (num_env, 1). Each element in the tensor indicates whether
    the unbalanced condition is active for the corresponding environment.

    Returns:
        torch.Tensor: A boolean tensor of shape (num_env, 1) where each value indicates whether
        the unbalanced termination condition is active (True) or not (False) for each environment.
    """
    # Apply threshold to the robot's projection of the gravity direction on base frame.
    abs_projected_gravity = torch.abs(projected_gravity)
    return torch.any(
        torch.logical_or(
            abs_projected_gravity[:, 0].unsqueeze(1) > gravity_x_threshold,
            abs_projected_gravity[:, 1].unsqueeze(1) > gravity_y_threshold,
        ),
        dim=1,
        keepdim=True,
    )


def terminate_by_undesired_contact(
    net_contact_forces: torch.Tensor, undesired_contact_body_ids: torch.Tensor
) -> torch.Tensor:
    """
    Checks termination condition based on contact forces.

    This function evaluates whether undesired bodies of the robot are in contact and returns a
    boolean tensor of shape (num_env, 1). Each element in the tensor indicates whether the contact
    termination condition is active for the corresponding environment.

    Args:
        net_contact_forces (torch.Tensor): A tensor of shape (num_env, num_bodies, 3).
        undesired_contact_body_ids (torch.Tensor): A tensor of shape (num_undesired_bodies,).

    Returns:
        torch.Tensor: A boolean tensor of shape (num_env, 1) where each value indicates whether
        the contact termination condition is active (True) or not (False) for each environment.
    """
    # Index 0 selects the most recent contact force in the history
    undesired_bodies_contact_forces = net_contact_forces[:, undesired_contact_body_ids]
    undesired_bodies_contact_forces = torch.norm(undesired_bodies_contact_forces, dim=-1)
    max_contact_force = torch.max(undesired_bodies_contact_forces, dim=1).values
    contact_force_threshold = 1.0
    return (max_contact_force > contact_force_threshold).view(-1, 1)


def terminate_by_reference_motion_length(
    ref_motion_mgr: ReferenceMotionManager, episode_times: torch.Tensor
) -> torch.Tensor:
    """Checks if the reference motion has ended for each environment.

    This function returns a boolean tensor of shape (num_env, 1), where each element indicates
    whether the reference motion has ended for the corresponding environment.

    Returns:
        torch.tensor: A boolean tensor of shape (num_env, 1) where `True` indicates that the episode should terminate.
    """
    return ref_motion_mgr.episodes_exceed_motion_length(episode_times=episode_times).unsqueeze(1)


def terminate_by_reference_motion_distance(
    training_mode: bool,
    body_state: BodyState,
    ref_motion_state: ReferenceMotionState,
    max_ref_motion_dist: float,
    in_recovery: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Determines if the distance between current body positions and reference motion positions exceeds the allowed threshold.

    Args:
        is_training (bool): Flag indicating if the system is in training mode.
        body_state (BodyState): Current state of the humanoid bodies.
        ref_motion_state (ReferenceMotionState): Reference motion state for the humanoid to track.
        max_ref_motion_dist (float): Maximum allowable distance between current and reference body positions.
        in_recovery (torch.Tensor | None): Boolean tensor (num_envs, 1) indicating if each environment is in recovery mode.
                                           If None, the recovery condition is ignored.

    Returns:
        torch.Tensor: Boolean tensor (num_envs, 1) indicating if the termination condition is met for each instance.
    """
    body_positions = body_state.body_pos_extend
    reference_positions = ref_motion_state.body_pos_extend
    num_bodies = body_positions.shape[1]
    body_mask = mask[:, :num_bodies]

    # Calculate the distance between current and reference positions
    distance = torch.norm(body_positions - reference_positions, dim=-1) * body_mask

    if training_mode:
        # Check if any distance exceeds the threshold
        exceeds_threshold = torch.any(distance > max_ref_motion_dist, dim=-1, keepdim=True)
    else:
        # Check if the mean distance exceeds the threshold
        mean_distance = distance.mean(dim=-1, keepdim=True)
        exceeds_threshold = torch.any(mean_distance > max_ref_motion_dist, dim=-1, keepdim=True)

    if in_recovery is None:
        return exceeds_threshold

    # If in recovery, ensure we only terminate if not in recovery mode
    return torch.logical_and(exceeds_threshold, ~in_recovery)
