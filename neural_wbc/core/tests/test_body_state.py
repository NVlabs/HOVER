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
import unittest

from neural_wbc.core.body_state import BodyState


def random_quaternions(n=1):
    u1, u2, u3 = torch.rand(n), torch.rand(n), torch.rand(n)
    q = torch.zeros((n, 4))
    q[:, 0] = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q[:, 1] = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q[:, 2] = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q[:, 3] = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)
    return q


class TestBodyState(unittest.TestCase):
    def setUp(self):
        self.num_envs = 2
        self.num_bodies = 3
        self.num_joints = 2
        self.num_extended_bodies = 1
        self.root_id = 0

        # Create random tensors for initial states
        self.body_pos = torch.rand(self.num_envs, self.num_bodies, 3)
        self.body_rot = random_quaternions(self.num_envs * self.num_bodies).view(self.num_envs, self.num_bodies, 4)
        self.body_lin_vel = torch.rand(self.num_envs, self.num_bodies, 3)
        self.body_ang_vel = torch.rand(self.num_envs, self.num_bodies, 3)
        self.joint_pos = torch.rand(self.num_envs, self.num_joints)
        self.joint_vel = torch.rand(self.num_envs, self.num_joints)

        # Create an instance of BodyState
        self.body_state = BodyState(
            body_pos=self.body_pos,
            body_rot=self.body_rot,
            body_lin_vel=self.body_lin_vel,
            body_ang_vel=self.body_ang_vel,
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
            root_id=self.root_id,
        )

    def test_initialization(self):
        self.assertTrue(torch.equal(self.body_state.body_pos, self.body_pos))
        self.assertTrue(torch.equal(self.body_state.body_rot, self.body_rot))
        self.assertTrue(torch.equal(self.body_state.body_lin_vel, self.body_lin_vel))
        self.assertTrue(torch.equal(self.body_state.body_ang_vel, self.body_ang_vel))
        self.assertTrue(torch.equal(self.body_state.joint_pos, self.joint_pos))
        self.assertTrue(torch.equal(self.body_state.joint_vel, self.joint_vel))

        # Without extension, body_*_extend is the same as body_*
        self.assertTrue(torch.equal(self.body_state.body_pos_extend, self.body_pos))
        self.assertTrue(torch.equal(self.body_state.body_rot_extend, self.body_rot))
        self.assertTrue(torch.equal(self.body_state.body_lin_vel_extend, self.body_lin_vel))
        self.assertTrue(torch.equal(self.body_state.body_ang_vel_extend, self.body_ang_vel))

    def test_extend_body_states(self):
        extend_body_pos = torch.rand(self.num_envs, self.num_extended_bodies, 3)
        extend_body_parent_ids = [0]

        self.body_state.extend_body_states(extend_body_pos, extend_body_parent_ids)

        self.assertEqual(
            self.body_state.body_pos_extend.shape, (self.num_envs, self.num_bodies + self.num_extended_bodies, 3)
        )
        self.assertEqual(
            self.body_state.body_rot_extend.shape, (self.num_envs, self.num_bodies + self.num_extended_bodies, 4)
        )
        self.assertEqual(
            self.body_state.body_lin_vel_extend.shape, (self.num_envs, self.num_bodies + self.num_extended_bodies, 3)
        )
        self.assertEqual(
            self.body_state.body_ang_vel_extend.shape, (self.num_envs, self.num_bodies + self.num_extended_bodies, 3)
        )


if __name__ == "__main__":
    unittest.main()
