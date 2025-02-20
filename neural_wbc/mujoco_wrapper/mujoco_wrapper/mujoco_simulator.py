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

import mujoco as mj
from mujoco_wrapper.mujoco_utils import get_entity_id, get_entity_name, has_free_joint
from mujoco_wrapper.utils import squeeze_if_tensor, to_numpy
from mujoco_wrapper.visualization import MujocoVisualizer

from neural_wbc.core.reference_motion import ReferenceMotionState


class WBCMujoco:
    """Whole Body Control using Mujoco Simulator

    High-level wrapper for the Mujoco environment for use in Whole Body Control (WBC).
    Provides a number of functionalities, including:
        - Get the ids, names of all joints and bodies in the model.
        - Get the joint positions, velocities.
        - Get the body positions, rotations, and velocities
        - Access to the underlying model and data
        - Step and reset the simulator based on control actions

    Example usage:
    .. code-block:: python
        wbc_mujoco = WBCMujoco('humanoid.xml')
        wbc_mujoco.reset()
        print(wbc_mujoco.joint_names)

    """

    def __init__(
        self,
        model_path: str,
        sim_dt: float = 0.005,
        enable_viewer: bool = False,
        num_instances: int = 1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        """Initialize the underlying mujoco simulator

        Args:
            model_path: Path to the Mujoco model xml file
            sim_dt: Simulation timestep
            enable_viewer: Whether to enable the viewer
            num_instances: Number of instances to simulate
            device: torch device to use for the tensors used by the simulator

        Raises:
            ValueError: If num_instances is not 1
        """
        assert num_instances == 1, "Only support a single instance for now."

        self._model = mj.MjModel.from_xml_path(model_path)
        self._model.opt.timestep = sim_dt
        self._data = mj.MjData(self._model)
        self.device = device
        self.has_free_joint = has_free_joint(self._model)
        self.joint_pos_offset = self._model.nq - self._model.nu  # Because positions are in generalized coordinates
        self.joint_vel_offset = self._model.nv - self._model.nu  # Because velocities include the free joint

        self.num_instances = num_instances

        self._viewer = None
        if enable_viewer:
            self._viewer = MujocoVisualizer(self._model, self._data)

        actuator_consistency = self._check_actuator_consistency()
        assert actuator_consistency, "Only support that all the actuator use the same control type."

    @property
    def model(self) -> mj.MjModel:
        """Get the underlying model

        Returns:
            MjModel: The underlying model
        """
        return self._model

    @property
    def data(self) -> mj.MjData:
        """Get the underlying data

        Returns:
            MjData: The underlying data
        """
        return self._data

    @property
    def joint_names(self) -> list[str]:
        """Get the names of all joints in the model except the free floating joint.

        Returns:
            list[str]: List of joint names
        """
        offset = 0
        if self.has_free_joint:
            offset = 1  # base/world joint
        return [get_entity_name(self._model, "joint", i) for i in range(offset, self._model.njnt)]

    @property
    def body_names(self) -> list[str]:
        """Get the names of all bodies in the model except the world body which is 0 always.

        Returns:
            list[str]: List of body names
        """
        return [get_entity_name(self._model, "body", i) for i in range(1, self._model.nbody)]

    @property
    def joint_positions(self) -> torch.Tensor:
        """Get the joint positions of the robot as tensor

        Returns:
            torch.Tensor: Tensor of joint positions
        """
        return (
            torch.from_numpy(self._data.qpos[self.joint_pos_offset :].copy())
            .to(dtype=torch.float32, device=self.device)
            .expand(self.num_instances, -1)
        )

    @property
    def joint_velocities(self) -> torch.Tensor:
        """Get the joint velocities of the robot as tensor

        Returns:
            torch.Tensor: Tensor of joint velocities
        """
        return (
            torch.from_numpy(self._data.qvel[self.joint_vel_offset :].copy())
            .to(dtype=torch.float32, device=self.device)
            .expand(self.num_instances, -1)
        )

    @property
    def body_positions(self) -> torch.Tensor:
        """Get the body positions of the robot as tensor

        Returns:
            torch.Tensor: Tensor of body positions
        """
        # NOTE: Global frame, https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
        # Get the body positions, excluding the first body (which is typically the world)
        robot_body_positions = torch.from_numpy(self._data.xpos[1:].copy())
        robot_body_positions = robot_body_positions.to(dtype=torch.float32, device=self.device)
        robot_body_positions = robot_body_positions.expand(self.num_instances, -1, -1)

        return robot_body_positions

    @property
    def body_rotations(self) -> torch.Tensor:
        """Get the body rotations of the robot as tensor

        Returns:
            torch.Tensor: Tensor of body rotations
        """
        # NOTE: Global frame, https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
        # Get the body positions, excluding the first body (which is typically the world)
        robot_body_rots = torch.from_numpy(self._data.xquat[1:].copy())
        robot_body_rots = robot_body_rots.to(dtype=torch.float32, device=self.device)
        robot_body_rots = robot_body_rots.expand(self.num_instances, -1, -1)

        return robot_body_rots

    @property
    def body_velocities(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the body linear and angular velocities of the robot as a pair of tensors

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of linear and angular body velocities
        """
        linear_velocities = torch.zeros(self.num_instances, self._model.nbody - 1, 3, device=self.device)
        angular_velocities = torch.zeros(self.num_instances, self._model.nbody - 1, 3, device=self.device)

        vel_frame = 0
        # NOTE the last parameter indicates the frame to use for velocity calculation, options are:
        # 0 - world frame, 1 - body frame, etc.
        for _, body_id in self.get_body_ids().items():
            # NOTE First three components are linear velocity, the next three are angular velocity
            vel_store = np.zeros(6)

            # Convert it back to the actual body_id inside the Mujoco model
            mj_body_id = body_id + 1
            mj.mj_objectVelocity(self._model, self._data, mj.mjtObj.mjOBJ_BODY, mj_body_id, vel_store, vel_frame)
            angular_velocities[:, body_id, :] = torch.from_numpy(vel_store[:3]).to(
                dtype=torch.float32, device=self.device
            )
            linear_velocities[:, body_id, :] = torch.from_numpy(vel_store[3:]).to(
                dtype=torch.float32, device=self.device
            )
        return linear_velocities, angular_velocities

    def forward(self) -> None:
        """Update the data structure of the mujoco model without stepping the simulator.

        This function should be called when terms in self._data are set manually and you would like the updated
        information to propagate to other related terms.

        Note that this function does not update the model, only the data.
        """
        mj.mj_forward(self._model, self._data)

    def step(self, actions: np.ndarray | None = None, nsteps: int = 1) -> None:
        """Step the simulation forward nsteps with the given action.

        Args:
            actions (np.ndarray | None, optional): Action to apply to the robot. Defaults to None.
            nsteps (int, optional): Number of steps to take. Defaults to 1.
        """

        if actions is None:
            actions = np.zeros((self.num_instances, self._model.nu))

        if actions.shape != (self.num_instances, self._model.nu):
            raise ValueError(
                f"Action shape {actions.shape} does not match number of actuators"
                f" {(self.num_instances, self._model.nu)}"
            )

        self._data.ctrl[:] = actions

        for _ in range(nsteps):
            mj.mj_step(self._model, self._data)
            self.update_viewer()

    def reset(
        self,
        qpos: np.ndarray | torch.Tensor | None = None,
        qvel: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """Reset the model to its initial state

        Args:
            qpos (np.ndarray | torch.Tensor | None, optional): Positions of the generalized coordinates. Defaults to None.
            qvel (np.ndarray | torch.Tensor | None, optional): Velocities of the generalized coordinates. Defaults to None.
        """
        mj.mj_resetData(self._model, self._data)

        self.set_robot_state(qpos, qvel)

    def set_robot_state(
        self, qpos: np.ndarray | torch.Tensor | None = None, qvel: np.ndarray | torch.Tensor | None = None
    ):
        """
        Set robot state including positions and velocities of the generalized coordinates.

        Args:
            qpos (np.ndarray | torch.Tensor | None, optional): Positions of the generalized coordinates. Defaults to None.
            qvel (np.ndarray | torch.Tensor | None, optional): Velocities of the generalized coordinates. Defaults to None.
        """
        if qpos is not None:
            # Ensure qpos length matches number of joints
            qpos = squeeze_if_tensor(qpos)
            qpos = to_numpy(qpos)
            assert len(qpos) == self._model.nq, f"qpos length {len(qpos)} doesn't match model DoF {self._model.nq}"
            self._data.qpos[:] = qpos

        if qvel is not None:
            # Ensure qvel length matches number of joints
            qvel = squeeze_if_tensor(qvel)
            qvel = to_numpy(qvel)
            assert len(qvel) == self._model.nv, f"qvel length {len(qvel)} doesn't match model DoF {self._model.nv}"
            self._data.qvel[:] = qvel

        # Forward to update robot state
        self.forward()

    def update_viewer(self) -> None:
        """Update the Mujoco viewer."""
        if self._viewer:
            self._viewer.update()

    def get_body_ids(self, body_names: list[str] | None = None, free_joint_offset: int = 1) -> dict[str, int]:
        """Get the IDs of all bodies in the model, indexed after removing the world body.

        Args:
            body_names (list[str] | None, optional): Names of the bodies. Defaults to None.
            free_joint_offset (int, optional): Offset to remove the free joint. Defaults to 1.

        Returns:
            dict[str, int]: Mapping from body name to body id.
        """
        body_names_ = body_names if body_names else self.body_names
        body_ids = {}
        for name in body_names_:
            id_ = get_entity_id(self._model, "body", name)
            if id_ > 0:
                body_ids[name] = id_ - free_joint_offset
            else:
                body_ids[name] = id_
        return body_ids

    def get_joint_ids(self, joint_names: list[str] | None = None, free_joint_offset: int = 1) -> dict[str, int]:
        """Get the IDs of all joints in the model, indexed after removing the free joint.

        Args:
            joint_names (list[str] | None, optional): Names of the joints. Defaults to None.
            free_joint_offset (int, optional): Offset to remove the free joint. Defaults to 1.

        Returns:
            dict[str, int]: Mapping from joint name to joint id.
        """
        joint_name_ = joint_names if joint_names else self.joint_names
        joint_ids = {}
        for name in joint_name_:
            id_ = get_entity_id(self._model, "joint", name)
            if id_ > 0:
                joint_ids[name] = id_ - free_joint_offset
            else:
                joint_ids[name] = id_
        return joint_ids

    def get_body_pose(self, body_name: str = "pelvis") -> tuple[torch.Tensor, torch.Tensor]:
        """Get the position and quaternion of the base

        Args:
            body_name (str, optional): Name of the body. Defaults to 'pelvis'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Position and quaternion of the base
        """
        body_id = self.get_body_ids([body_name])[body_name]
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in the model.")
        # Convert it back to the actual body_id inside the Mujoco model
        body_id += 1

        body_pos = (
            torch.from_numpy(self._data.xpos[body_id].copy()).to(device=self.device).expand(self.num_instances, -1)
        )
        body_quat = (
            torch.from_numpy(self._data.xquat[body_id].copy()).to(device=self.device).expand(self.num_instances, -1)
        )

        return body_pos, body_quat

    def get_base_projected_gravity(self, base_name: str = "pelvis") -> torch.Tensor:
        """Get the projection of the gravity vector to the base frame

        Args:
            base_name (str, optional): Name of the base. Defaults to 'pelvis'.

        Returns:
            torch.Tensor: Projection of the gravity vector to the base frame
        """
        world_gravity = self._model.opt.gravity
        # Normalize the gravity to match IsaacLab.
        world_gravity = world_gravity / np.linalg.norm(world_gravity)

        body_id = self.get_body_ids([base_name])[base_name]
        if body_id < 0:
            raise ValueError(f"Base '{base_name}' not found in the model.")
        # Convert it back to the actual body_id inside the Mujoco model
        body_id += 1

        root_b_w = np.linalg.inv(self.data.xmat[body_id].reshape(3, 3))
        grav = root_b_w @ world_gravity  # Gravity vector in body frame

        return torch.tensor(grav, device=self.device, dtype=torch.float32).expand(self.num_instances, -1)

    def get_terrain_heights(self) -> torch.Tensor:
        """Get the terrain height.

        Note: before the actual terrain height sensor needs to be added to the model, we assume they are just zeros.

        Returns:
            torch.Tensor: Terrain height
        """
        return torch.zeros(self.num_instances, 1).to(dtype=torch.float32, device=self.device)

    def get_contact_forces_with_floor(self, body_name: str) -> torch.Tensor:
        """Get the contact forces on a given body with the floor

        Args:
            body_name (str): Name of the body

        Returns:
            torch.Tensor: Contact forces, shape (num_envs, 3), i.e. normal and two tangent directions

        Notes:
            Only checks contacts with the floor, and thus assumes the loaded model contains a floor geometry object.
            This can be easily ensured by loading the scene.xml file that includes the specific robot model xml file.
        """
        if body_name not in self.body_names:
            raise ValueError(f"Body '{body_name}' not found in the model.")

        zero_contact = torch.zeros(self.num_instances, 3).to(dtype=torch.float32, device=self.device)
        if self._data.ncon == 0:
            return zero_contact

        body_id = self.get_body_ids([body_name])[body_name]
        if body_id < 0:
            raise ValueError(f"Base '{body_name}' not found in the model.")
        body_id += 1

        # NOTE: We assume existence of a floor and only check the contacts with the floor
        floor_body_id = 0

        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            contact_body_id_1 = self._model.geom_bodyid[contact.geom1]
            contact_body_id_2 = self._model.geom_bodyid[contact.geom2]
            if {contact_body_id_1, contact_body_id_2} == {body_id, floor_body_id}:
                contact_force = np.zeros(6)
                mj.mj_contactForce(self._model, self._data, i, contact_force)
                return torch.from_numpy(contact_force[:3].copy()).to(device=self.device).expand(self.num_instances, -1)

        return zero_contact

    def print_actuator_info(self, actuators_id: list[int] | None = None):
        """Utility function to print out actuator types in the model.

        Args:
            actuators_id (list[int] | None, optional): Actuator ids. Defaults to None.
        """
        actuators_id_ = range(self.model.nu) if actuators_id is None else actuators_id
        for actuator_id in actuators_id_:
            print(f"Actuator {actuator_id}:")
            # Print meaning of ctrl for this actuator
            if self.model.actuator_gaintype[actuator_id] == mj.mjtGain.mjGAIN_FIXED:
                print("Direct force/torque control")
            elif self.model.actuator_gaintype[actuator_id] == mj.mjtGain.mjGAIN_AFFINE:
                if self.model.actuator_biastype[actuator_id] == mj.mjtBias.mjBIAS_NONE:
                    print("Force/torque control with scaling")
                else:
                    print("Position or velocity control")

    def visualize_ref_state(self, ref_motion_state: ReferenceMotionState):
        """Visualize the reference motion state in Mujoco viewer, if enabled

        Args:
            ref_motion_state (ReferenceMotionState): Reference motion state
        """
        if self._viewer:
            self._viewer.draw_reference_state(ref_motion_state)

    def _check_actuator_consistency(self):
        """Check whether all the actuators share the same control mode."""
        actuator_type_system = None
        for actuator_id in range(self.model.nu):
            actuator_type = self.model.actuator_trntype[actuator_id]
            if actuator_type_system is None:
                actuator_type_system = actuator_type
            else:
                if actuator_type_system != actuator_type:
                    return False

        return True
