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
from neural_wbc.core.mask import calculate_mask_length
from neural_wbc.core.reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg, ReferenceMotionState
from neural_wbc.core.termination import (
    check_termination_conditions,
    terminate_by_gravity,
    terminate_by_reference_motion_distance,
    terminate_by_reference_motion_length,
    terminate_by_undesired_contact,
)
from neural_wbc.data import get_data_path


class TestTerminationConditions(unittest.TestCase):
    def setUp(self):
        self.num_envs = 2
        self.num_bodies = 3
        self.num_joints = 2
        self.root_id = 0
        self.mask_length = calculate_mask_length(self.num_bodies, self.num_joints)

        self.body_state = BodyState(
            body_pos=torch.rand(self.num_envs, self.num_bodies, 3),
            body_rot=torch.rand(self.num_envs, self.num_bodies, 4),
            body_lin_vel=torch.rand(self.num_envs, self.num_bodies, 3),
            body_ang_vel=torch.rand(self.num_envs, self.num_bodies, 3),
            joint_pos=torch.rand(self.num_envs, self.num_joints),
            joint_vel=torch.rand(self.num_envs, self.num_joints),
            root_id=self.root_id,
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
            "dof_pos": torch.rand(self.num_envs, 2),
            "dof_vel": torch.rand(self.num_envs, 2),
        }
        self.ref_motion_state = ReferenceMotionState(ref_motion)

        self.projected_gravity = torch.tensor([[0.2, 0.15, 0.9], [0.6, 0.1, 0.8]])
        self.gravity_x_threshold = 0.5
        self.gravity_y_threshold = 0.2

        cfg = ReferenceMotionManagerCfg()
        cfg.motion_path = get_data_path("motions/stable_punch.pkl")
        cfg.skeleton_path = get_data_path("motion_lib/h1.xml")
        self.ref_motion_mgr = ReferenceMotionManager(
            cfg=cfg,
            device=torch.device("cpu"),
            num_envs=self.num_envs,
            random_sample=False,
            extend_head=False,
            dt=0.005,
        )
        self.episode_times = torch.tensor([0.0, 11.0])
        self.max_ref_motion_dist = 0.5
        self.in_recovery = torch.tensor([[False], [True]])
        self.net_contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3)
        self.net_contact_forces[0, 0, 0] = 10.0  # Env 0, Body 0 is in contact.
        self.undesired_contact_body_ids = torch.tensor([0, 1])
        self.mask = torch.ones(self.num_envs, self.mask_length).bool()

        # Per default the body positions are all within the tolerance of the max_ref_motion_distance
        self.body_state.body_pos_extend = self.ref_motion_state.body_pos_extend + 0.1 * self.max_ref_motion_dist

    def test_terminate_by_gravity(self):
        termination = terminate_by_gravity(
            projected_gravity=self.projected_gravity,
            gravity_x_threshold=self.gravity_x_threshold,
            gravity_y_threshold=self.gravity_y_threshold,
        )
        expected = torch.tensor([[False], [True]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

    def test_terminate_by_undesired_contact(self):
        termination = terminate_by_undesired_contact(
            net_contact_forces=self.net_contact_forces,
            undesired_contact_body_ids=self.undesired_contact_body_ids,
        )
        expected = torch.tensor([[True], [False]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

    def test_terminate_by_reference_motion_length(self):
        termination = terminate_by_reference_motion_length(
            ref_motion_mgr=self.ref_motion_mgr, episode_times=self.episode_times
        )
        expected = torch.tensor([[False], [True]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

    def test_terminate_by_reference_motion_distance_training_mode(self):
        # Per default all body positions are within the limits.
        termination = terminate_by_reference_motion_distance(
            training_mode=True,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            max_ref_motion_dist=self.max_ref_motion_dist,
            in_recovery=self.in_recovery,
            mask=self.mask,
        )
        expected = torch.tensor([[False], [False]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

        # Now we move body 0 in all envs out of bounds. In training mode we expect to terminate if
        # any body is out of bounds. We only expect env 0 to terminate because env 1 is in recovery
        # mode.
        self.body_state.body_pos_extend[:, 0, :] = (
            self.ref_motion_state.body_pos_extend[:, 0, :]
            + torch.ones_like(self.ref_motion_state.body_pos_extend[:, 0, :]) * self.max_ref_motion_dist
        )
        termination = terminate_by_reference_motion_distance(
            training_mode=True,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            max_ref_motion_dist=self.max_ref_motion_dist,
            in_recovery=self.in_recovery,
            mask=self.mask,
        )
        expected = torch.tensor([[True], [False]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

        # Now we move all bodies in all envs out of bounds. We still only expect env 0 to terminate
        # because env 1 is in recovery mode.
        self.body_state.body_pos_extend = (
            self.ref_motion_state.body_pos_extend
            + torch.ones_like(self.ref_motion_state.body_pos_extend) * self.max_ref_motion_dist
        )
        termination = terminate_by_reference_motion_distance(
            training_mode=True,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            max_ref_motion_dist=self.max_ref_motion_dist,
            in_recovery=self.in_recovery,
            mask=self.mask,
        )
        expected = torch.tensor([[True], [False]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

        # Now we set the mask to all 0. Thus all bodies should be ignored and no env should
        # terminate.
        mask = torch.zeros_like(self.mask)
        termination = terminate_by_reference_motion_distance(
            training_mode=True,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            max_ref_motion_dist=self.max_ref_motion_dist,
            in_recovery=self.in_recovery,
            mask=mask,
        )
        expected = torch.tensor([[False], [False]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

    def test_terminate_by_reference_motion_distance_non_training_mode(self):
        # Per default all body positions are within the limits.
        termination = terminate_by_reference_motion_distance(
            training_mode=False,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            max_ref_motion_dist=self.max_ref_motion_dist,
            in_recovery=self.in_recovery,
            mask=self.mask,
        )
        expected = torch.tensor([[False], [False]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

        # Now we move body 0 in all envs out of bounds. In non-training mode we expect to terminate
        # if the mean of all bodies is out of bounds. Thus we expect no termination.
        self.body_state.body_pos_extend[:, 0, :] = (
            self.ref_motion_state.body_pos_extend[:, 0, :]
            + torch.ones_like(self.ref_motion_state.body_pos_extend[:, 0, :]) * self.max_ref_motion_dist
        )
        termination = terminate_by_reference_motion_distance(
            training_mode=False,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            max_ref_motion_dist=self.max_ref_motion_dist,
            in_recovery=self.in_recovery,
            mask=self.mask,
        )
        expected = torch.tensor([[False], [False]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

        # Now we move all bodies in all envs out of bounds, thus the mean is no out of bounds. We
        # still only expect env 0 to terminate because env 1 is in recovery mode.
        self.body_state.body_pos_extend = (
            self.ref_motion_state.body_pos_extend
            + torch.ones_like(self.ref_motion_state.body_pos_extend) * self.max_ref_motion_dist
        )
        termination = terminate_by_reference_motion_distance(
            training_mode=False,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            max_ref_motion_dist=self.max_ref_motion_dist,
            in_recovery=self.in_recovery,
            mask=self.mask,
        )
        expected = torch.tensor([[True], [False]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

        # Now we set the mask to all 0. Thus all bodies should be ignored and no env should
        # terminate.
        mask = torch.zeros_like(self.mask)
        termination = terminate_by_reference_motion_distance(
            training_mode=False,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            max_ref_motion_dist=self.max_ref_motion_dist,
            in_recovery=self.in_recovery,
            mask=mask,
        )
        expected = torch.tensor([[False], [False]])
        self.assertTrue(torch.equal(termination, expected), f"termination: {termination}, expected: {expected}")

    def test_check_termination_conditions(self):
        should_terminate, conditions = check_termination_conditions(
            training_mode=True,
            projected_gravity=self.projected_gravity,
            gravity_x_threshold=self.gravity_x_threshold,
            gravity_y_threshold=self.gravity_y_threshold,
            ref_motion_mgr=self.ref_motion_mgr,
            episode_times=self.episode_times,
            body_state=self.body_state,
            ref_motion_state=self.ref_motion_state,
            max_ref_motion_dist=self.max_ref_motion_dist,
            in_recovery=self.in_recovery,
            net_contact_forces=self.net_contact_forces,
            undesired_contact_body_ids=self.undesired_contact_body_ids,
            mask=self.mask,
        )
        expected_should_terminate = torch.tensor([True, True])
        expected_conditions = {
            "gravity": torch.tensor([[False], [True]]),
            "undesired_contact": torch.tensor([[True], [False]]),
            "reference_motion_length": torch.tensor([[False], [True]]),
            "reference_motion_distance": torch.tensor([[False], [False]]),
        }
        self.assertTrue(torch.equal(should_terminate, expected_should_terminate))
        for key in expected_conditions:
            self.assertTrue(torch.equal(conditions[key], expected_conditions[key]))


if __name__ == "__main__":
    unittest.main()
