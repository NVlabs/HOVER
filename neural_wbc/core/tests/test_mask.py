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

from neural_wbc.core.mask import calculate_mask_length, create_mask, create_mask_element_names


class TestMask(unittest.TestCase):
    def test_mask_helpers(self):
        body_names = ["a", "b", "c"]
        joint_names = ["d", "e", "f"]
        mask_length = calculate_mask_length(len(body_names), len(joint_names))
        mask_element_names = create_mask_element_names(body_names, joint_names)
        self.assertEqual(mask_length, len(mask_element_names))

    def test_single_mode_mask(self):
        num_envs = 4
        mask_element_names = ["a", "b", "c", "d", "e", "f"]
        mask_modes = {
            "mode_1": {
                "group_a": ["a", "b", "c"],
            }
        }
        mask = create_mask(
            num_envs=num_envs,
            mask_element_names=mask_element_names,
            mask_modes=mask_modes,
            enable_sparsity_randomization=False,
            device=torch.device("cpu"),
        )

        # Check basic properties
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, (num_envs, len(mask_element_names)))

        # All environments should have the same mask since there's only one mode
        self.assertTrue(torch.all(mask[:, :3]))  # First three elements should be True
        self.assertTrue(torch.all(~mask[:, 3:]))  # Last three elements should be False

    def test_multiple_modes_mask(self):
        num_envs = 100  # Large number to ensure we see both modes
        mask_element_names = ["a", "b", "c", "d", "e", "f"]
        mask_modes = {
            "mode_1": {
                "group_a": ["a", "b"],
            },
            "mode_2": {
                "group_b": ["c", "d"],
            },
        }
        mask = create_mask(
            num_envs=num_envs,
            mask_element_names=mask_element_names,
            mask_modes=mask_modes,
            enable_sparsity_randomization=False,
            device=torch.device("cpu"),
        )

        # Check that we have both modes present
        mode1_pattern = torch.tensor([True, True, False, False, False, False])
        mode2_pattern = torch.tensor([False, False, True, True, False, False])

        has_mode1 = False
        has_mode2 = False
        for env_idx in range(num_envs):
            if torch.all(mask[env_idx] == mode1_pattern):
                has_mode1 = True
            if torch.all(mask[env_idx] == mode2_pattern):
                has_mode2 = True

        self.assertTrue(has_mode1 and has_mode2, "Both modes should be present with enough environments")

    def test_sparsity_randomization(self):
        num_envs = 100
        mask_element_names = ["a", "b", "c", "d", "e", "f"]
        mask_modes = {
            "mode_1": {
                "group_a": ["a", "b", "c"],
            }
        }
        mask = create_mask(
            num_envs=num_envs,
            mask_element_names=mask_element_names,
            mask_modes=mask_modes,
            enable_sparsity_randomization=True,
            device=torch.device("cpu"),
        )

        # With sparsity randomization, we expect:
        # 1. No True values in the last three elements
        self.assertTrue(torch.all(~mask[:, 3:]))

        # 2. Some variation in the first three elements
        first_three = mask[:, :3]
        self.assertTrue(
            torch.any(first_three) and not torch.all(first_three), "Sparsity randomization should create some variation"
        )

        # 3. Average should be roughly 0.5 for the active elements
        mean_active = first_three.float().mean()
        self.assertTrue(0.4 <= mean_active <= 0.6, f"Mean of active elements should be close to 0.5, got {mean_active}")

    def test_overlapping_patterns(self):
        num_envs = 4
        mask_element_names = ["a", "b", "c", "d"]
        mask_modes = {"mode_1": {"group_a": ["a", "b"], "group_b": ["b", "c"]}}  # Overlapping 'b'
        mask = create_mask(
            num_envs=num_envs,
            mask_element_names=mask_element_names,
            mask_modes=mask_modes,
            enable_sparsity_randomization=False,
            device=torch.device("cpu"),
        )

        # Check that overlapping elements are handled correctly
        expected_pattern = torch.tensor([True, True, True, False])
        self.assertTrue(torch.all(mask == expected_pattern.unsqueeze(0)))


if __name__ == "__main__":
    unittest.main()
