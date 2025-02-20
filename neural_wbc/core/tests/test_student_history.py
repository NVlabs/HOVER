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

from neural_wbc.core.observations import StudentHistory


class TestStudentHistory(unittest.TestCase):
    def setUp(self):
        self.num_envs = 5
        self.entry_length = 6
        self.max_entries = 5
        self.device = torch.device("cpu")

        self.history = StudentHistory(self.num_envs, self.device, self.entry_length, self.max_entries)

    def test_initialization(self):
        self.assertEqual(self.history.entries.shape, (self.num_envs, self.entry_length * self.max_entries))
        self.assertEqual(self.history._entry_length, self.entry_length)
        self.assertEqual(self.history._entry_shape, torch.Size([self.num_envs, self.entry_length]))

    def test_update_correct_shape(self):
        obs_dict = {
            "distilled_robot_state": torch.rand(self.num_envs, 4),
            "distilled_last_action": torch.rand(self.num_envs, 2),
        }
        self.history.update(obs_dict)
        self.assertTrue(
            torch.equal(
                self.history.entries[:, : self.entry_length],
                torch.cat([obs_dict["distilled_robot_state"], obs_dict["distilled_last_action"]], dim=-1),
            )
        )

    def test_update_incorrect_shape(self):
        obs_dict = {
            "distilled_robot_state": torch.rand(self.num_envs, 3),
            "distilled_last_action": torch.rand(self.num_envs, 2),
        }
        with self.assertRaises(AssertionError):
            self.history.update(obs_dict)

    def test_update_missing_key(self):
        obs_dict = {
            "distilled_robot_state": torch.rand(self.num_envs, 4),
            # "distilled_last_action" key is missing
        }
        with self.assertRaises(KeyError):
            self.history.update(obs_dict)

    def test_history_rollover(self):
        for _ in range(self.max_entries + 1):
            obs_dict = {
                "distilled_robot_state": torch.rand(self.num_envs, 4),
                "distilled_last_action": torch.rand(self.num_envs, 2),
            }
            self.history.update(obs_dict)

        self.assertEqual(self.history.entries.shape, (self.num_envs, self.entry_length * self.max_entries))

    def test_reset_all(self):
        self.history._entries = torch.rand_like(self.history.entries)
        self.history.reset(None)
        self.assertTrue(torch.equal(self.history.entries, torch.zeros_like(self.history.entries)))

    def test_reset_specific_envs(self):
        self.history._entries = torch.rand_like(self.history.entries)
        env_ids = torch.tensor([1, 3])
        self.history.reset(env_ids)
        for env_id in env_ids:
            self.assertTrue(torch.equal(self.history.entries[env_id], torch.zeros_like(self.history.entries[env_id])))
        for env_id in set(range(self.num_envs)) - set(env_ids.tolist()):
            self.assertFalse(torch.equal(self.history.entries[env_id], torch.zeros_like(self.history.entries[env_id])))


if __name__ == "__main__":
    unittest.main()
