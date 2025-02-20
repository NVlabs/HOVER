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

from neural_wbc.core.robot_wrapper import Robot, register_robot

from .mujoco_simulator import WBCMujoco


@register_robot
class MujocoRobot(Robot):
    """Simulated UniTree H1 robot in Mujoco"""

    def __init__(
        self,
        cfg: Any,
        num_instances: int = 1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__(cfg, num_instances=num_instances, device=device)

        self._sim = WBCMujoco(
            model_path=cfg.model_xml_path,
            sim_dt=cfg.dt,
            enable_viewer=cfg.enable_viewer,
            num_instances=num_instances,
            device=device,
        )

        self._joint_names = self._sim.joint_names
        self._body_names = self._sim.body_names

        self._free_joint_offset = 1 if self._sim.has_free_joint else 0

        self.joint_pos_offset = self._sim.joint_pos_offset
        self.joint_vel_offset = self._sim.joint_vel_offset

    def update(self, obs_dict: dict[str, torch.Tensor | list[str] | None]) -> None:
        """Update the underlying model based on the observations from the environment/real robot.

        Args:
            obs_dict (dict[str, torch.Tensor]): A dictionary containing the latest robot observations.
        """
        if "root_pos" in obs_dict:
            self._root_position = obs_dict["root_pos"]
        if "root_orientation" in obs_dict:
            self._root_rotation = obs_dict["root_orientation"]

        self._joint_positions = self._sim.joint_positions
        self._joint_velocities = self._sim.joint_velocities
        self._body_positions = self._sim.body_positions
        self._body_rotations = self._sim.body_rotations
        self._body_lin_vels, self._body_ang_vels = self._sim.body_velocities

    def reset(self, **kwargs) -> None:
        """Resets the wrapper

        Args:
            kwargs (dict[str, Any], optional): key-word arguments to pass to underlying models. Defaults to None.
        """
        qpos = kwargs.get("qpos")
        qvel = kwargs.get("qvel")
        self._sim.reset(qpos=qpos, qvel=qvel)
        self.update({})

    def visualize(self, **payload) -> None:
        """Visualize some info

        Args:
            payload (dict[str, Any], optional): key-word arguments to pass to underlying models. Defaults to None.

        """
        if "ref_motion_state" in payload:
            self._sim.visualize_ref_state(payload["ref_motion_state"])

    def step(self, actions: np.ndarray | None = None, nsteps: int = 1) -> None:
        """Step the simulation forward nsteps with the given action.

        Args:
            actions (np.ndarray | None, optional): Action to apply to the robot. Defaults to None.
            nsteps (int, optional): Number of steps to take. Defaults to 1.
        """
        self._sim.step(actions, nsteps)

    def get_body_ids(self, body_names: list[str] | None = None) -> dict[str, int]:
        """Get the IDs of all bodies in the model, indexed after removing the world body.

        Args:
            body_names (list[str] | None, optional): Names of the bodies. Defaults to None.

        Returns:
            dict[str, int]: Mapping from body name to body id.
        """
        return self._sim.get_body_ids(body_names, self._free_joint_offset)

    def get_joint_ids(self, joint_names: list[str] | None = None) -> dict[str, int]:
        """Get the IDs of all joints in the model, indexed after removing the free joint.

        Args:
            joint_names (list[str] | None, optional): Names of the joints. Defaults to None.

        Returns:
            dict[str, int]: Mapping from joint name to joint id.
        """
        return self._sim.get_joint_ids(joint_names, self._free_joint_offset)

    def get_body_pose(self, body_name: str = "pelvis") -> tuple[torch.Tensor, torch.Tensor]:
        """Get the position and quaternion of the base

        Args:
            body_name (str, optional): Name of the body. Defaults to 'pelvis'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Position and quaternion of the base
        """
        return self._sim.get_body_pose(body_name)

    def get_base_projected_gravity(self, base_name: str = "pelvis") -> torch.Tensor:
        """Get the projection of the gravity vector to the base frame

        Args:
            base_name (str, optional): Name of the base. Defaults to 'pelvis'.

        Returns:
            torch.Tensor: Projection of the gravity vector to the base frame
        """
        return self._sim.get_base_projected_gravity(base_name)

    def get_terrain_heights(self) -> torch.Tensor:
        """Get the terrain height.

        Note: before the actual terrain height sensor needs to be added to the model, we assume they are just zeros.

        Returns:
            torch.Tensor: Terrain height
        """
        return self._sim.get_terrain_heights()

    @property
    def internal_sim(self):
        """Gets the internal MJC simulator instance"""
        return self._sim
