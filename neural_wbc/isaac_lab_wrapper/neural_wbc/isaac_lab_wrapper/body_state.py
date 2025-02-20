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

from neural_wbc.core.body_state import BodyState

from isaaclab.assets import ArticulationData


def build_body_state(
    data: ArticulationData,
    root_id: int,
    body_ids: list[int] | None = None,
    joint_ids: list[int] | None = None,
    extend_body_pos: torch.Tensor | None = None,
    extend_body_parent_ids: list[int] | None = None,
) -> BodyState:
    """Creates a body state from Isaac Lab articulation data.

    Args:
        data (ArticulationData): Articulation data containing robot's body and joint states.
        body_ids (list[int] | None, optional): The desired order of bodies. If not, the order in body states is preserved.
                                               Defaults to None.
        joint_ids (list[int] | None, optional): The desired order of joint. If not, the order in joint states is preserved.
                                                Defaults to None.
        extend_body_parent_ids (list[int] | None, optional): ID of the bodies to extend. Defaults to None.
        extend_body_pos (torch.Tensor | None, optional): Position of the extended bodies from their parent bodies. Defaults to None.

    Returns:
        BodyState: The constructed BodyState object containing the reordered and extended body and joint states.
    """
    if body_ids is None:
        num_bodies = data.body_pos_w.shape[1]
        body_ids = list(range(0, num_bodies))

    if joint_ids is None:
        num_joints = data.joint_pos.shape[1]
        joint_ids = list(range(0, num_joints))

    body_state = BodyState(
        body_pos=data.body_pos_w[:, body_ids, :],
        body_rot=data.body_quat_w[:, body_ids, :],
        body_lin_vel=data.body_lin_vel_w[:, body_ids, :],
        body_ang_vel=data.body_ang_vel_w[:, body_ids, :],
        joint_pos=data.joint_pos[:, joint_ids],
        joint_vel=data.joint_vel[:, joint_ids],
        root_id=root_id,
    )

    if (extend_body_pos is not None) and (extend_body_parent_ids is not None):
        body_state.extend_body_states(extend_body_pos=extend_body_pos, extend_body_parent_ids=extend_body_parent_ids)

    return body_state
