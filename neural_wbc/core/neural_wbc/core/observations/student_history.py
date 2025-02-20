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


class StudentHistory:
    def __init__(self, num_envs: int, device: torch.device, entry_length: int, max_entries: int):
        """
        Args:
            env (EnvironmentWrapper): An instance of the environment wrapper.
            entry_length (int): The length of a single entry.
            max_entries (int): The maximum number of entries to keep in the history.
        """
        self._entries = torch.zeros(
            num_envs,
            entry_length * max_entries,
            device=device,
        )
        self._entry_length = entry_length
        self._entry_shape = torch.Size([num_envs, entry_length])

    @property
    def entries(self):
        return self._entries.clone()

    def update(self, obs_dict: dict[str, torch.Tensor]):
        """Updates the history with a new entry.

        Args:
            obs_dict (dict[str, torch.Tensor]): A dictionary containing the latest student observations.
                Expected keys are "distilled_robot_state" and "distilled_last_action".

        Raises:
            AssertionError: If the new entry does not match the expected shape.
            KeyError: If the observation does not contain expected keys.
        """
        new_entry = torch.cat(
            [obs_dict["distilled_robot_state"], obs_dict["distilled_last_action"]],
            dim=-1,
        )
        assert new_entry.shape == self._entry_shape, "New entry has an unexpected shape."
        self._entries[:, self._entry_length :] = self._entries[:, : -self._entry_length].clone()
        self._entries[:, : self._entry_length] = new_entry.clone()

    def reset(self, env_ids: torch.Tensor | None):
        """Resets the history for specified environments.

        Args:
            env_ids (torch.Tensor | None): A tensor containing the IDs of the environments to reset.
                                           If None, all environments are reset.

        Example:
            history.reset(None)  # Resets history for all environments
            history.reset(torch.tensor([0, 2]))  # Resets history for environments with IDs 0 and 2
        """
        if env_ids is None:
            self._entries[:] = 0.0
        self._entries[env_ids, ...] = 0.0
