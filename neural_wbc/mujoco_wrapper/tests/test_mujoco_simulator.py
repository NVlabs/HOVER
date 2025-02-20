# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import unittest

import mujoco as mj
from mujoco_wrapper.mujoco_simulator import WBCMujoco
from mujoco_wrapper.mujoco_utils import get_entity_id, get_entity_name

from neural_wbc.data import get_data_path


class TestWBCMujoco(unittest.TestCase):
    """Tests for WBCMujoco wrapper"""

    def setUp(self):
        # Create a WBCMujoco object for testing
        # NOTE: We load the scene.xml to include the floor used for contact forces testing
        self.model_path = get_data_path("mujoco/models/scene.xml")
        self.sim_dt = 0.005
        self.enable_viewer = False
        self.device = torch.device("cpu")
        self.wbc_mj = WBCMujoco(
            model_path=self.model_path,
            sim_dt=self.sim_dt,
            enable_viewer=self.enable_viewer,
            device=self.device,
        )

    def test_joint_names(self):
        # Test the joint_names property
        expected_joint_names = [
            get_entity_name(self.wbc_mj.model, "joint", i)
            for i in range(1, self.wbc_mj.model.njnt)  # Skip the free joint
        ]
        self.assertEqual(self.wbc_mj.joint_names, expected_joint_names)

    def test_body_names(self):
        # Test the body_names property
        expected_body_names = [get_entity_name(self.wbc_mj.model, "body", i) for i in range(1, self.wbc_mj.model.nbody)]
        self.assertEqual(self.wbc_mj.body_names, expected_body_names)

    def test_get_body_ids(self):
        # Test the get_body_ids method
        expected_body_ids = {
            name: get_entity_id(self.wbc_mj.model, "body", name) - 1 for name in self.wbc_mj.body_names
        }
        all_body_id_names = self.wbc_mj.get_body_ids()
        sorted_body_ids = sorted(all_body_id_names.values())
        self.assertEqual(sorted_body_ids[0], 0)
        self.assertEqual(all_body_id_names, expected_body_ids)

        specific_body_names = ["pelvis", "torso_link"]
        expected_body_ids = {name: get_entity_id(self.wbc_mj.model, "body", name) - 1 for name in specific_body_names}
        self.assertEqual(
            self.wbc_mj.get_body_ids(body_names=specific_body_names),
            expected_body_ids,
        )

        non_existent_body_names = ["nonexistent_body"]
        non_existent_body_ids = self.wbc_mj.get_body_ids(body_names=non_existent_body_names)
        self.assertEqual(non_existent_body_ids, {name: -1 for name in non_existent_body_names})

    def test_get_joint_ids(self):
        # Test the get_joint_ids method
        expected_joint_ids = {
            name: get_entity_id(self.wbc_mj.model, "joint", name) - 1 for name in self.wbc_mj.joint_names
        }
        all_joint_id_names = self.wbc_mj.get_joint_ids()
        sorted_joint_ids = sorted(all_joint_id_names.values())
        self.assertEqual(sorted_joint_ids[0], 0)
        self.assertEqual(all_joint_id_names, expected_joint_ids)

        specific_joint_names = ["left_elbow", "torso"]
        expected_joint_ids = {
            name: get_entity_id(self.wbc_mj.model, "joint", name) - 1 for name in specific_joint_names
        }
        self.assertEqual(
            self.wbc_mj.get_joint_ids(joint_names=specific_joint_names),
            expected_joint_ids,
        )

    def test_joint_positions(self):
        # Test the joint_positions property
        expected_joint_positions = (
            torch.from_numpy(self.wbc_mj.data.qpos[self.wbc_mj.joint_pos_offset :].copy())
            .to(dtype=torch.float32, device=self.device)
            .expand(self.wbc_mj.num_instances, -1)
        )

        self.assertTrue(torch.allclose(self.wbc_mj.joint_positions, expected_joint_positions))

    def test_joint_velocities(self):
        # Test the joint_velocities property
        expected_joint_velocities = (
            torch.from_numpy(self.wbc_mj.data.qvel[self.wbc_mj.joint_vel_offset :].copy())
            .to(dtype=torch.float32, device=self.device)
            .expand(self.wbc_mj.num_instances, -1)
        )

        self.assertTrue(torch.allclose(self.wbc_mj.joint_velocities, expected_joint_velocities))

    def test_body_positions(self):
        # Test the body_positions property
        expected_body_positions = (
            torch.from_numpy(self.wbc_mj.data.xpos[1:].copy())
            .to(dtype=torch.float32, device=self.device)
            .expand(self.wbc_mj.num_instances, -1, -1)
        )

        self.assertTrue(torch.allclose(self.wbc_mj.body_positions, expected_body_positions))

    def test_body_rotations(self):
        # Test the body_rotations property
        expected_body_rotations = (
            torch.from_numpy(self.wbc_mj.data.xquat[1:].copy())
            .to(dtype=torch.float32, device=self.device)
            .expand(self.wbc_mj.num_instances, -1, -1)
        )

        self.assertTrue(torch.allclose(self.wbc_mj.body_rotations, expected_body_rotations))

    def test_body_velocity_indexing(self):
        expected_all_body_vels = []
        for body_id in range(self.wbc_mj.model.nbody):
            vel_store = np.zeros(6)
            mj.mj_objectVelocity(self.wbc_mj.model, self.wbc_mj.data, mj.mjtObj.mjOBJ_BODY, body_id, vel_store, 0)
            expected_all_body_vels.append(vel_store)

        expected_all_body_vels = torch.tensor(expected_all_body_vels, dtype=torch.float)
        # Exclude the world body.
        expected_robot_body_vels = expected_all_body_vels[1:, :]

        lin_vel, ang_vel = self.wbc_mj.body_velocities
        self.assertTrue(torch.allclose(expected_robot_body_vels[:, :3], ang_vel))
        self.assertTrue(torch.allclose(expected_robot_body_vels[:, 3:], lin_vel))

    def test_forward(self):
        # Test the forward method
        jp_before_forward = self.wbc_mj.joint_positions.clone()

        self.wbc_mj.forward()

        self.assertTrue(torch.allclose(self.wbc_mj.joint_positions, jp_before_forward))

    def test_step(self):
        # Set the qpos and qvel to non-zero numbers for valid check.
        rand_qpos = np.random.rand(*self.wbc_mj.data.qpos.shape)
        rand_qpos[self.wbc_mj.joint_pos_offset :] = np.random.rand(1, self.wbc_mj.model.nu)
        rand_qvel = np.random.rand(*self.wbc_mj.data.qvel.shape)
        self.wbc_mj.set_robot_state(qpos=rand_qpos, qvel=rand_qvel)

        # Test the visualize_ref_state method
        controls = torch.zeros(self.wbc_mj.num_instances, self.wbc_mj.model.nu, device=self.device)
        actions = controls.detach().cpu().numpy()

        jp_before_step = self.wbc_mj.joint_positions.clone()

        self.wbc_mj.step(actions=actions)

        self.assertFalse(torch.allclose(self.wbc_mj.joint_positions, jp_before_step))

    def test_get_contact_forces(self):
        current_qpos = torch.from_numpy(self.wbc_mj.data.qpos).to(dtype=torch.float, device=self.device).expand(1, -1)
        current_qpos[:, :3] -= 0.1
        self.wbc_mj.set_robot_state(qpos=current_qpos)

        # First the robot is NOT floating, so the contact forces should NOT be zero.
        contact_forces = self.wbc_mj.get_contact_forces_with_floor("right_ankle_link")
        self.assertEqual(contact_forces.shape, (self.wbc_mj.num_instances, 3))
        self.assertFalse(torch.allclose(contact_forces, torch.zeros_like(contact_forces)))

        # Now the robot IS floating, so the contact forces should be zero.
        current_qpos = torch.from_numpy(self.wbc_mj.data.qpos).to(dtype=torch.float, device=self.device).expand(1, -1)
        base_pose = torch.tensor([0.0, 0.0, 2.5])
        current_qpos[:, :3] = base_pose
        self.wbc_mj.set_robot_state(qpos=current_qpos)

        new_contact_forces = self.wbc_mj.get_contact_forces_with_floor("right_ankle_link")
        self.assertTrue(torch.allclose(new_contact_forces, torch.zeros_like(new_contact_forces)))

    def test_set_robot_state(self):
        # Set the base position
        current_qpos = torch.from_numpy(self.wbc_mj.data.qpos).to(dtype=torch.float, device=self.device).expand(1, -1)
        base_pose = torch.tensor([0.0, 0.0, 2.5])
        current_qpos[:, :3] = base_pose
        self.wbc_mj.set_robot_state(qpos=current_qpos)
        qpos_after_set = torch.from_numpy(self.wbc_mj.data.qpos).to(dtype=torch.float, device=self.device).expand(1, -1)
        self.assertTrue(torch.allclose(qpos_after_set[:, :3], base_pose))

        # Set the joint positions
        current_qpos = torch.from_numpy(self.wbc_mj.data.qpos).to(dtype=torch.float, device=self.device).expand(1, -1)
        new_joint_positions = torch.rand((self.wbc_mj.num_instances, self.wbc_mj.model.nu))
        current_qpos[:, 7:] = new_joint_positions
        self.wbc_mj.set_robot_state(qpos=current_qpos)
        qpos_after_set = torch.from_numpy(self.wbc_mj.data.qpos).to(dtype=torch.float, device=self.device).expand(1, -1)
        self.assertTrue(torch.allclose(qpos_after_set[:, 7:], new_joint_positions))


if __name__ == "__main__":
    unittest.main()
