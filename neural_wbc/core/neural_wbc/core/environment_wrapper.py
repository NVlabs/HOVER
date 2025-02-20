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

from .modes import NeuralWBCModes
from .reference_motion import ReferenceMotionManager


class EnvironmentWrapper:
    """Class that provides an interface to perform training and evaluation in different simulators."""

    num_envs: int  # Number of environments running in the simulator
    device: torch.device
    reference_motion_manager: ReferenceMotionManager

    def __init__(self, mode: NeuralWBCModes):
        self._mode = mode

    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """Performs one step of simulation.
        Returns:
        * Observations as a dict or an object (TBD).
        * Rewards of shape (num_envs,)
        * "Dones": a boolean tensor representing termination of an episode in each environment.
        * Extra information captured, as a dict.
        """
        raise NotImplementedError

    def reset(self, env_ids: list | torch.Tensor):
        """Resets environment specified by env_ids."""
        raise NotImplementedError

    def get_observations(self) -> torch.Tensor:
        """Gets policy observations for each environment based on the mode."""
        if self._mode.is_distill_mode():
            return self.get_student_observations()
        return self.get_teacher_observations()

    def get_teacher_observations(self) -> torch.Tensor:
        """Gets teacher policy observations for each environment."""
        raise NotImplementedError

    def get_student_observations(self) -> torch.Tensor:
        """Gets student policy observations for each environment."""
        raise NotImplementedError
