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

from neural_wbc.student_policy.storage import Slice, Storage


class TestStorage(unittest.TestCase):
    """
    Unit tests for the Storage class.

    This module contains unit tests to validate the functionality of the Storage class.
    All test values, including numbers and shapes, are arbitrary and do not correspond
    to any specific type of robots or real-world scenarios.
    """

    def test_add(self):
        max_steps = 10
        num_envs = 3
        device = torch.device("cpu")
        policy_obs_shape = torch.Size([19, 3])
        student_obs_shape = torch.Size([9, 3])
        actions_shape = torch.Size([10])

        storage = Storage(
            max_steps=max_steps,
            num_envs=num_envs,
            device=device,
            policy_obs_shape=policy_obs_shape,
            student_obs_shape=student_obs_shape,
            actions_shape=actions_shape,
        )
        self.assertEqual(storage.step, 0)

        policy_observations = torch.randn(max_steps, num_envs, *policy_obs_shape)
        student_observations = torch.randn(max_steps, num_envs, *student_obs_shape)
        ground_truth_actions = torch.randn(max_steps, num_envs, *actions_shape)
        student_actions = torch.randn(max_steps, num_envs, *actions_shape)

        for step in range(max_steps):
            slice = Slice(
                policy_observations=policy_observations[step, ...],
                student_observations=student_observations[step, ...],
                ground_truth_actions=ground_truth_actions[step, ...],
                applied_actions=student_actions[step, ...],
            )
            storage.add(slice)
            self.assertEqual(storage.step, step + 1)

        self.assertTrue(torch.all(torch.eq(storage.policy_observations, policy_observations)))
        self.assertTrue(torch.all(torch.eq(storage.student_observations, student_observations)))
        self.assertTrue(torch.all(torch.eq(storage.ground_truth_actions, ground_truth_actions)))
        self.assertTrue(torch.all(torch.eq(storage.applied_actions, student_actions)))

        self.assertRaises(AssertionError, storage.add, slice)

        storage.reset()
        self.assertEqual(storage.step, 0)


if __name__ == "__main__":
    unittest.main()
