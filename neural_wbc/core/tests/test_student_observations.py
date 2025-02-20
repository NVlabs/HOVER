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
from neural_wbc.core.mask import calculate_command_length, calculate_mask_length
from neural_wbc.core.observations.student_observations import (
    compute_distilled_imitation_observations,
    compute_distilled_robot_state_observation,
    compute_student_observations,
)
from neural_wbc.core.reference_motion import ReferenceMotionState


class TestStudentObservations(unittest.TestCase):
    def setUp(self):
        self.num_envs = 2
        self.num_bodies = 3
        self.num_joints = 2
        self.mask_length = calculate_mask_length(self.num_bodies, self.num_joints)
        self.command_length = calculate_command_length(self.num_bodies, self.num_joints)
        self.base_id = 0

        self.body_state = BodyState(
            body_pos=torch.rand(self.num_envs, self.num_bodies, 3),
            body_rot=torch.rand(self.num_envs, self.num_bodies, 4),
            body_lin_vel=torch.rand(self.num_envs, self.num_bodies, 3),
            body_ang_vel=torch.rand(self.num_envs, self.num_bodies, 3),
            joint_pos=torch.rand(self.num_envs, self.num_joints),
            joint_vel=torch.rand(self.num_envs, self.num_joints),
            root_id=self.base_id,
        )

        ref_motion = {
            "root_pos": torch.rand(self.num_envs, 3),
            "root_rot": torch.rand(self.num_envs, 4),
            "root_vel": torch.rand(self.num_envs, 3),
            "root_ang_vel": torch.rand(self.num_envs, 3),
            "rg_pos": torch.rand(self.num_envs, self.num_bodies, 3),
            "rb_rot": torch.rand(self.num_envs, self.num_bodies, 4),
            "body_vel": torch.rand(self.num_envs, self.num_bodies, 3),
            "body_ang_vel": torch.rand(self.num_envs, self.num_bodies, 3),
            "rg_pos_t": torch.rand(self.num_envs, self.num_bodies, 3),
            "rg_rot_t": torch.rand(self.num_envs, self.num_bodies, 4),
            "body_vel_t": torch.rand(self.num_envs, self.num_bodies, 3),
            "body_ang_vel_t": torch.rand(self.num_envs, self.num_bodies, 3),
            "dof_pos": torch.rand(self.num_envs, self.num_joints),
            "dof_vel": torch.rand(self.num_envs, self.num_joints),
        }
        self.ref_motion_state = ReferenceMotionState(ref_motion)
        self.projected_gravity = torch.rand(self.num_envs, 3)
        self.last_actions = torch.rand(self.num_envs, 2)
        self.history = torch.rand(self.num_envs, 10)
        self.ref_episodic_offset = torch.rand(self.num_envs, 3)
        self.mask = torch.ones(self.num_envs, self.mask_length).bool()

    def test_compute_distilled_robot_state_observation(self):
        obs = compute_distilled_robot_state_observation(
            body_state=self.body_state, base_id=self.base_id, projected_gravity=self.projected_gravity
        )
        expected_shape = (self.num_envs, self.body_state.joint_pos.shape[1] + self.body_state.joint_vel.shape[1] + 6)
        self.assertEqual(obs.shape, expected_shape)

    def test_compute_distilled_imitation_observations(self):
        obs = compute_distilled_imitation_observations(
            ref_motion_state=self.ref_motion_state,
            body_state=self.body_state,
            ref_episodic_offset=self.ref_episodic_offset,
            mask=self.mask,
        )
        expected_shape = (self.num_envs, self.command_length + self.mask_length)
        self.assertEqual(obs.shape, expected_shape)

    def test_compute_student_observations(self):
        obs, obs_dict = compute_student_observations(
            base_id=self.base_id,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            projected_gravity=self.projected_gravity,
            last_actions=self.last_actions,
            history=self.history,
            ref_episodic_offset=self.ref_episodic_offset,
            mask=self.mask,
        )

        expected_shape = (
            self.num_envs,
            # Proprioceptive states:
            self.body_state.joint_pos.shape[1]
            + self.body_state.joint_vel.shape[1]
            + 3  # local base angular velocity
            + 3  # projected gravity
            # Command:
            + self.command_length
            # Mask:
            + self.mask_length
            + +self.last_actions.shape[1]  # Last actions:
            + +self.history.shape[1],  # Historical info
        )
        self.assertEqual(obs.shape, expected_shape)
        self.assertIn("distilled_robot_state", obs_dict)
        self.assertIn("distilled_imitation", obs_dict)
        self.assertIn("distilled_last_action", obs_dict)
        self.assertIn("distilled_historical_info", obs_dict)


if __name__ == "__main__":
    unittest.main()
