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
from dataclasses import dataclass

import phc.utils.torch_utils as torch_utils

from neural_wbc.core import math_utils


@dataclass
class BodyState:
    """
    Encapsulates data of humanoid bodies and joints in all simulation instances.

    Note: The body_*_extend attributes represent states after the body is extended.

    Attributes
    ----------
    body_pos : torch.Tensor
        Positions of all bodies in world frame. Shape is (num_envs, num_bodies, 3).
    body_rot : torch.Tensor
        Orientation of all bodies in world frame as quaternions, with convention wxyz.
        Shape is (num_envs, num_bodies, 4).
    body_lin_vel : torch.Tensor
        Linear velocities of all bodies in world frame. Shape is (num_envs, num_bodies, 3).
    body_ang_vel : torch.Tensor
        Angular velocities of all bodies in world frame. Shape is (num_envs, num_bodies, 3).
    body_pos_extend : torch.Tensor
        Positions of all extended bodies in world frame. Shape is (num_envs, num_bodies, 3).
    body_rot_extend : torch.Tensor
        Orientation of all extended bodies in world frame. Shape is (num_envs, num_bodies, 4).
    body_lin_vel_extend : torch.Tensor
        Linear velocities of all extended bodies in world frame. Shape is (num_envs, num_bodies, 3).
    body_ang_vel_extend : torch.Tensor
        Angular velocities of all extended bodies in world frame. Shape is (num_envs, num_bodies, 3).
    joint_pos : torch.Tensor
        Joint positions of all joints. Shape is (num_envs, num_joints).
    joint_vel : torch.Tensor
        Joint velocities of all joints. Shape is (num_envs, num_joints).
    """

    body_pos: torch.Tensor
    body_rot: torch.Tensor
    body_lin_vel: torch.Tensor
    body_ang_vel: torch.Tensor

    body_pos_extend: torch.Tensor
    body_rot_extend: torch.Tensor
    body_lin_vel_extend: torch.Tensor
    body_ang_vel_extend: torch.Tensor

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor

    def __init__(
        self,
        body_pos: torch.Tensor,
        body_rot: torch.Tensor,
        body_lin_vel: torch.Tensor,
        body_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_id: int,
    ):
        self.body_pos = body_pos.clone()
        self.body_rot = body_rot.clone()
        self.body_lin_vel = body_lin_vel.clone()
        self.body_ang_vel = body_ang_vel.clone()

        self.body_pos_extend = body_pos.clone()
        self.body_rot_extend = body_rot.clone()
        self.body_lin_vel_extend = body_lin_vel.clone()
        self.body_ang_vel_extend = body_ang_vel.clone()

        self.joint_pos = joint_pos.clone()
        self.joint_vel = joint_vel.clone()

        self.root_id = root_id

    @property
    def root_pos(self) -> torch.Tensor:
        return self.body_pos[:, self.root_id, :]

    @property
    def root_rot(self) -> torch.Tensor:
        return self.body_rot[:, self.root_id, :]

    @property
    def root_lin_vel(self) -> torch.Tensor:
        return self.body_lin_vel[:, self.root_id, :]

    @property
    def root_ang_vel(self) -> torch.Tensor:
        return self.body_ang_vel[:, self.root_id, :]

    def extend_body_states(
        self,
        extend_body_pos: torch.Tensor,
        extend_body_parent_ids: list[int],
    ):
        """
        This function is for appending the link states to the robot state. For example, the H1 robot doesn't have hands
        and a head in its robot state. However, we are still interested in computing its error and considering these as
        important key points. Thus, we will use this function to add the head and hand states to the robot state.

        Args:
            extend_body_pos (torch.Tensor): Positions of the extended bodies relative to their parent bodies.
                Shape is (num_envs, num_extended_bodies, 3).
            extend_body_parent_ids (list[int]): List of parent body indices for each extended body.

        Raises:
            ValueError: If the number of extended bodies does not match the length of extend_body_parent_ids.
        """
        if extend_body_pos.shape[1] != len(extend_body_parent_ids):
            print("[INFO]: extend_body_pos shape:", extend_body_pos.shape)
            print("[INFO]: extend_body_parent_ids lengths:", len(extend_body_parent_ids))
            raise ValueError(
                "Dimension mismatch: number of extended bodies does not match the length of its parent ID list."
            )

        num_envs = self.body_pos.shape[0]

        # Compute extended body positions
        extend_curr_pos = (
            torch_utils.my_quat_rotate(
                math_utils.convert_quat(self.body_rot[:, extend_body_parent_ids].reshape(-1, 4), to="xyzw"),
                extend_body_pos[:,].reshape(-1, 3),
            ).view(num_envs, -1, 3)
            + self.body_pos[:, extend_body_parent_ids]
        )
        self.body_pos_extend = torch.cat([self.body_pos, extend_curr_pos], dim=1)

        # Compute extended body orientations
        extend_curr_rot = self.body_rot[:, extend_body_parent_ids].clone()
        self.body_rot_extend = torch.cat([self.body_rot, extend_curr_rot], dim=1)

        # Compute extended body linear velocities
        self.body_lin_vel_extend = torch.cat(
            [self.body_lin_vel, self.body_lin_vel[:, extend_body_parent_ids].clone()], dim=1
        )

        # Compute extended body angular velocities
        self.body_ang_vel_extend = torch.cat(
            [self.body_ang_vel, self.body_ang_vel[:, extend_body_parent_ids].clone()], dim=1
        )
