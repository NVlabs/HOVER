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


import numpy as np
import torch
import unittest

from mujoco_wrapper.mujoco_simulator import WBCMujoco
from mujoco_wrapper.mujoco_utils import get_entity_id, get_entity_name
from mujoco_wrapper.utils import squeeze_if_tensor, to_numpy

from neural_wbc.data import get_data_path


class TestMujocoBodyState(unittest.TestCase):
    def setUp(self):
        self.num_instances = 1
        self.device = torch.device("cpu")
        self.robot = WBCMujoco(model_path=get_data_path("mujoco/models/h1.xml"), device=self.device)

    def test_body_state_data(self):
        num_instances = self.num_instances
        num_controls = self.robot.model.nu
        # Confirm sizes are correct
        self.assertEqual(self.robot.joint_positions.shape, torch.Size((num_instances, num_controls)))
        self.assertEqual(self.robot.joint_velocities.shape, torch.Size((num_instances, num_controls)))

        # Only consider the robot bodies.
        num_bodies = len(self.robot.body_names)
        self.assertEqual(self.robot.body_positions.shape, torch.Size((num_instances, num_bodies, 3)))  # [x,y,z]
        self.assertEqual(self.robot.body_rotations.shape, torch.Size((num_instances, num_bodies, 4)))  # [w,x,y,z]

        lin_vel, ang_vel = self.robot.body_velocities
        self.assertEqual(lin_vel.shape, torch.Size((num_instances, num_bodies, 3)))
        self.assertEqual(ang_vel.shape, torch.Size((num_instances, num_bodies, 3)))


class TestMujocoUtils(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.robot = WBCMujoco(model_path=get_data_path("mujoco/models/h1.xml"), device=self.device)

        self.sample_h1_body_ids = {
            "pelvis": 0,
            "left_hip_yaw_link": 1,
            "left_hip_roll_link": 2,
            "left_hip_pitch_link": 3,
            "left_knee_link": 4,
            "left_ankle_link": 5,
        }

    def test_free_joint_check(self):
        self.assertEqual(self.robot.has_free_joint, True)

    def test_get_entity_name(self):
        # body
        test_body_id = 0
        output_body_name = get_entity_name(self.robot.model, "body", test_body_id + 1)
        self.assertEqual(output_body_name, "pelvis")

        # Missing element
        self.assertRaises(IndexError, lambda: get_entity_name(self.robot.model, "body", -1))

    def test_get_entity_id(self):
        # body
        test_body_name = "pelvis"
        expected_body_id = 1  # NOTE: We mirror the transformation in WBCMujoco here
        output_body_id = get_entity_id(self.robot.model, "body", test_body_name)
        self.assertEqual(output_body_id, expected_body_id)

        # Missing element
        self.assertEqual(get_entity_id(self.robot.model, "body", "random_name"), -1)
        self.assertEqual(get_entity_id(self.robot.model, "joint", "random_name"), -1)

    def test_to_numpy(self):
        # Test with a PyTorch tensor
        x = torch.tensor([1, 2, 3])
        expected_output = np.array([1, 2, 3])
        output = to_numpy(x)
        self.assertTrue(np.allclose(output, expected_output))

        # Test with a numpy array
        x = np.array([1, 2, 3])
        expected_output = np.array([1, 2, 3])
        output = to_numpy(x)
        self.assertTrue(np.allclose(output, expected_output))

    def test_squeeze_if_tensor(self):
        # Test with a PyTorch tensor
        x = torch.tensor([[1], [2], [3]])
        expected_output = torch.tensor([1, 2, 3])
        output = squeeze_if_tensor(x, dim=1)
        self.assertTrue(torch.allclose(output, expected_output))

        # Test with a numpy array
        x = np.array([[1], [2], [3]])
        expected_output = np.array([1, 2, 3])
        output = squeeze_if_tensor(x, dim=1)
        self.assertFalse(np.allclose(output, expected_output))
