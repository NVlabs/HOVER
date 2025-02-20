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

import numpy as np
import torch
from typing import Any


class Robot:
    """Robot base for doing inference with HOVER and OmniH2O.

    This provides a interface layer for different robot platforms (simulated or real), used
    to do inference with a HOVER or OmniH2O policies.

    Depending on the use case, the user may provide an `obs_dict` that already contains all
    the required information such as body poses and joint positions. In this case, the
    wrapper just need to update the corresponding internal items.

    In other cases, the user can define a internal kinematic model, then only need to provide the
    root information and the joint states to update the internal model. The other kinematic terms
    are then correspondingly obtained via the update the model.

    Note:
        Using a internal kinematic model may prevent from running on GPU. For example, the
        plain Mujoco model can only run on CPU.

    Example:
    .. code-block:: python
        robot = RobotWrapper()
        robot.update(obs_dict)
    """

    def __init__(
        self,
        cfg: Any,
        num_instances: int = 1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        """Initialize the underlying mujoco simulator"""
        self.num_controls = cfg.num_joints
        self.num_instances = num_instances
        self.device = device

        self._joint_names: list[str] = []
        self._joint_positions: torch.Tensor = torch.zeros(num_instances, self.num_controls, device=self.device)
        self._joint_velocities: torch.Tensor = torch.zeros(num_instances, self.num_controls, device=self.device)

        self._body_names: list[str] = []
        self._body_positions: torch.Tensor = torch.zeros(
            num_instances, len(self._body_names), 3, device=self.device
        )  # [x,y,z]
        self._body_rotations: torch.Tensor = torch.zeros(
            num_instances, len(self._body_names), 4, device=self.device
        )  # [w,x,y,z]
        self._body_ang_vels: torch.Tensor = torch.zeros(num_instances, len(self._body_names), 3, device=self.device)
        self._body_lin_vels: torch.Tensor = torch.zeros(num_instances, len(self._body_names), 3, device=self.device)

        # NOTE: add space for the root position (3) and orientation(4)
        self._default_joint_positions: torch.Tensor = torch.zeros(
            num_instances, self.num_controls + 7, device=self.device
        )
        # NOTE: add space for the root lin vel (3) and root ang vel(3)
        self._default_joint_velocities: torch.Tensor = torch.zeros(
            num_instances, self.num_controls + 6, device=self.device
        )

        self._root_position: torch.Tensor = torch.zeros(num_instances, 3, device=self.device)
        self._root_rotation: torch.Tensor = torch.zeros(num_instances, 4, device=self.device)
        self._root_lin_vel: torch.Tensor = torch.zeros(num_instances, 3, device=self.device)
        self._root_ang_vel: torch.Tensor = torch.zeros(num_instances, 3, device=self.device)
        self.joint_pos_offset = 0
        self.joint_vel_offset = 0

    def update(self, obs_dict: dict[str, torch.Tensor]) -> None:
        """Update the underlying model based on the observations from the environment/real robot.

        Args:
            obs_dict (dict[str, torch.Tensor]): A dictionary containing the latest robot observations.
        """
        raise NotImplementedError

    def reset(self, **kwargs) -> None:
        """Resets the wrapper

        Args:
            kwargs (dict[str, Any], optional): key-word arguments to pass to underlying models. Defaults to None.
        """
        raise NotImplementedError

    def visualize(self, **payload) -> None:
        """Visualize some info

        Args:
            payload (dict[str, Any], optional): key-word arguments to pass to underlying models. Defaults to None.

        """
        raise NotImplementedError

    def step(self, actions: np.ndarray | None = None, nsteps: int = 1) -> None:
        """Step the simulation forward nsteps with the given action.

        Args:
            actions (np.ndarray | None, optional): Action to apply to the robot. Defaults to None.
            nsteps (int, optional): Number of steps to take. Defaults to 1.
        """
        raise NotImplementedError

    def get_body_ids(self, body_names: list[str] | None = None) -> dict[str, int]:
        """Get the IDs of all bodies in the model, indexed after removing the world body.

        Args:
            body_names (list[str] | None, optional): Names of the bodies. Defaults to None.

        Returns:
            dict[str, int]: Mapping from body name to body id.
        """
        raise NotImplementedError

    def get_joint_ids(self, joint_names: list[str] | None = None) -> dict[str, int]:
        """Get the IDs of all joints in the model, indexed after removing the free joint.

        Args:
            joint_names (list[str] | None, optional): Names of the joints. Defaults to None.

        Returns:
            dict[str, int]: Mapping from joint name to joint id.
        """
        raise NotImplementedError

    def get_body_pose(self, body_name: str = "pelvis") -> tuple[torch.Tensor, torch.Tensor]:
        """Get the position and quaternion of the base

        Args:
            body_name (str, optional): Name of the body. Defaults to 'pelvis'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Position and quaternion of the base
        """
        raise NotImplementedError

    def get_base_projected_gravity(self, base_name: str = "pelvis") -> torch.Tensor:
        """Get the projection of the gravity vector to the base frame

        Args:
            base_name (str, optional): Name of the base. Defaults to 'pelvis'.

        Returns:
            torch.Tensor: Projection of the gravity vector to the base frame
        """
        raise NotImplementedError

    def get_terrain_heights(self) -> torch.Tensor:
        """Get the terrain height.

        Note: before the actual terrain height sensor needs to be added to the model, we assume they are just zeros.

        Returns:
            torch.Tensor: Terrain height
        """
        raise NotImplementedError

    @property
    def joint_names(self) -> list[str]:
        """Get the names of all joints in the model except the free floating joint.

        Returns:
            list[str]: List of joint names
        """
        return self._joint_names

    @property
    def body_names(self) -> list[str]:
        """Get the names of all bodies in the model except the world body which is 0 always.

        Returns:
            list[str]: List of body names
        """
        return self._body_names

    @property
    def joint_positions(self) -> torch.Tensor:
        """Get the joint positions of the robot as tensor

        Returns:
            torch.Tensor: Tensor of joint positions
        """
        return self._joint_positions

    @property
    def joint_velocities(self) -> torch.Tensor:
        """Get the joint velocities of the robot as tensor

        Returns:
            torch.Tensor: Tensor of joint velocities
        """
        return self._joint_velocities

    @property
    def body_positions(self) -> torch.Tensor:
        """Get the body positions of the robot as tensor

        Returns:
            torch.Tensor: Tensor of body positions
        """
        return self._body_positions

    @property
    def body_rotations(self) -> torch.Tensor:
        """Get the body rotations of the robot as tensor

        Returns:
            torch.Tensor: Tensor of body rotations
        """
        return self._body_rotations

    @property
    def body_velocities(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the body linear and angular velocities of the robot as a pair of tensors

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of linear and angular body velocities
        """
        return self._body_lin_vels, self._body_ang_vels

    @property
    def default_joint_positions(self) -> torch.Tensor:
        """Get the default joint positions of the robot as tensor

        Returns:
            torch.Tensor: Tensor of joint positions
        """
        return self._default_joint_positions

    @property
    def default_joint_velocities(self) -> torch.Tensor:
        """Get the default joint velocities of the robot as tensor

        Returns:
            torch.Tensor: Tensor of joint velocities
        """
        return self._default_joint_velocities


ROBOTS = {}

import re


def to_snake(input_str: str):
    """Converts input string to snake_case"""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", input_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def register_robot(cls):
    ROBOTS[to_snake(cls.__name__)] = cls
    return cls


def get_robot_names():
    return ROBOTS.keys()


def get_robot_class(name: str) -> type[Robot]:
    return ROBOTS[name]
