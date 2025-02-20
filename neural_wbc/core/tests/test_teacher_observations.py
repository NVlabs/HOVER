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
from neural_wbc.core.observations.teacher_observations import (
    compute_imitation_observations,
    compute_robot_state_observation,
    compute_teacher_observations,
)
from neural_wbc.core.reference_motion import ReferenceMotionState


class TestTeacherObservations(unittest.TestCase):
    def setUp(self):
        self.num_envs = 2
        self.num_bodies = 3
        self.num_joints = 2
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
        self.tracked_body_ids = [0, 1, 2]
        self.last_actions = torch.rand(self.num_envs, 2)
        self.ref_episodic_offset = torch.rand(self.num_envs, 3)

    def test_compute_robot_state_observation(self):
        obs = compute_robot_state_observation(body_state=self.body_state)
        expected_shape = (
            self.num_envs,
            (self.num_bodies - 1) * 3  # local_body_pos_obs without the root
            + self.num_bodies * 6  # local_body_rot_obs
            + self.num_bodies * 3  # local_body_vel
            + self.num_bodies * 3,  # local_body_ang_vel
        )
        self.assertEqual(obs.shape, expected_shape)

    def test_compute_imitation_observations(self):
        obs = compute_imitation_observations(
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            tracked_body_ids=self.tracked_body_ids,
            ref_episodic_offset=self.ref_episodic_offset,
        )
        expected_shape = (
            self.num_envs,
            len(self.tracked_body_ids) * 3  # diff_local_body_pos_flat
            + len(self.tracked_body_ids) * 6  # diff_local_body_rot_flat
            + len(self.tracked_body_ids) * 3  # diff_local_vel
            + len(self.tracked_body_ids) * 3  # diff_local_ang_vel
            + len(self.tracked_body_ids) * 3  # local_ref_body_pos
            + len(self.tracked_body_ids) * 6,  # local_ref_body_rot
        )
        self.assertEqual(obs.shape, expected_shape)

    def test_compute_teacher_observations(self):
        obs, obs_dict = compute_teacher_observations(
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            tracked_body_ids=self.tracked_body_ids,
            last_actions=self.last_actions,
            ref_episodic_offset=self.ref_episodic_offset,
        )
        expected_shape = (
            self.num_envs,
            (self.num_bodies - 1) * 3  # local_body_pos_obs without the root
            + self.num_bodies * 6  # local_body_rot_obs
            + self.num_bodies * 3  # local_body_vel
            + self.num_bodies * 3  # local_body_ang_vel
            + len(self.tracked_body_ids) * 3  # diff_local_body_pos_flat
            + len(self.tracked_body_ids) * 6  # diff_local_body_rot_flat
            + len(self.tracked_body_ids) * 3  # diff_local_vel
            + len(self.tracked_body_ids) * 3  # diff_local_ang_vel
            + len(self.tracked_body_ids) * 3  # local_ref_body_pos
            + len(self.tracked_body_ids) * 6  # local_ref_body_rot
            + self.last_actions.shape[1],  # last_actions
        )
        self.assertEqual(obs.shape, expected_shape)
        self.assertIn("robot_state", obs_dict)
        self.assertIn("imitation", obs_dict)
        self.assertIn("last_action", obs_dict)


if __name__ == "__main__":
    unittest.main()
