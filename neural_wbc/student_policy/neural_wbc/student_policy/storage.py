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
from dataclasses import dataclass


@dataclass
class Slice:
    policy_observations: torch.Tensor
    student_observations: torch.Tensor
    ground_truth_actions: torch.Tensor
    applied_actions: torch.Tensor


@dataclass
class Storage:
    step: int
    policy_observations: torch.Tensor
    student_observations: torch.Tensor
    ground_truth_actions: torch.Tensor
    applied_actions: torch.Tensor

    def __init__(
        self,
        max_steps: int,
        num_envs: int,
        device: torch.device,
        policy_obs_shape,
        student_obs_shape,
        actions_shape,
    ):
        self._max_steps = max_steps
        self._num_envs = num_envs
        self._device = device
        self._policy_obs_shape = policy_obs_shape
        self._student_obs_shape = student_obs_shape
        self._actions_shape = actions_shape

        self.reset()

    def is_full(self):
        return self.step >= self._max_steps

    def add(self, slice: Slice):
        if self.is_full():
            raise AssertionError("Rollout buffer overflow")
        self.policy_observations[self.step].copy_(slice.policy_observations)
        self.student_observations[self.step].copy_(slice.student_observations)
        self.ground_truth_actions[self.step].copy_(slice.ground_truth_actions)
        self.applied_actions[self.step].copy_(slice.applied_actions)
        self.step += 1

    def reset(self):
        self.step = 0
        self.policy_observations = torch.zeros(
            self._max_steps, self._num_envs, *self._policy_obs_shape, device=self._device
        )
        self.student_observations = torch.zeros(
            self._max_steps, self._num_envs, *self._student_obs_shape, device=self._device
        )
        self.ground_truth_actions = torch.zeros(
            self._max_steps, self._num_envs, *self._actions_shape, device=self._device
        )
        self.applied_actions = torch.zeros(self._max_steps, self._num_envs, *self._actions_shape, device=self._device)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self._num_envs * self._max_steps
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self._device)

        policy_observations = self.policy_observations.flatten(0, 1)
        student_observations = self.student_observations.flatten(0, 1)
        ground_truth_actions = self.ground_truth_actions.flatten(0, 1)
        applied_actions = self.applied_actions.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                policy_observations_batch = policy_observations[batch_idx]
                student_observations_batch = student_observations[batch_idx]
                ground_truth_actions_batch = ground_truth_actions[batch_idx]
                applied_actions_batch = applied_actions[batch_idx]

                yield policy_observations_batch, student_observations_batch, ground_truth_actions_batch, applied_actions_batch
