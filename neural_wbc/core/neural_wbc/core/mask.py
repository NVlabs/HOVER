# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .util import get_matching_indices


def calculate_mask_length(num_bodies: int, num_joints: int, num_root_references: int = 7) -> int:
    """
    Calculate the length of the mask that can be used to select which parts of the reference
    motion should be tracked. We assume that the robot is floating base.
    """
    return num_bodies + num_joints + num_root_references


def calculate_command_length(num_bodies: int, num_joints: int, num_root_references: int = 7) -> int:
    """
    Calculate the length of the reference motion command. This is different than the mask length
    because every body has 3 reference commands (x/y/z). We assume that the robot is floating base.
    """
    return num_bodies * 3 + num_joints + num_root_references


def create_mask_element_names(body_names: list[str], joint_names: list[str]):
    """Get a name for each element of the mask."""
    body_names = [name + "_local_pos_" for name in body_names]
    joint_names = [name + "_joint_pos" for name in joint_names]
    root_reference_names = [
        "root_linear_velocity_x",
        "root_linear_velocity_y",
        "root_linear_velocity_z",
        "root_orientation_roll",
        "root_orientation_pitch",
        "root_orientation_yaw_delta",
        "root_height",
    ]
    return body_names + joint_names + root_reference_names


def create_mask(
    num_envs: int,
    mask_element_names: list[str],
    mask_modes: dict[str, dict[str, list[str]]],
    enable_sparsity_randomization: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a mask where all enabled states are set to 1.
    This mask can be used directly or multiplied with 0.5 and then be used as the probability of
    a state being enabled.

    Args:
        mask_element_names: The name corresponding to every element in the mask.
        mask_modes: A nested dictionary configuring which mask elements are enabled in which mode.
            The nested dictionary takes the form `{mode_name: {body_region: [element_names]}}`
            The `mode_name` and `body_region` can be chosen freely, they are just for documentary
            value. The `element_names` may also use regex patterns.
            An example configuration is:

            "exbody": {
                "upper_body": [".*torso_joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
                "lower_body": ["root.*"],
            },

            For every `mode_name`, all elements of all body regions in that mode will be enabled.
            Most likely you want the elements specified in one body region to not
            overlap with other body regions (the code will still work if they do overlap, but the
            resulting trained policy may not work when the overlapping elements are disabled at test
            time).

        enable_sparsity_randomization: If enabled random elements of the mask will be disabled with
            probability 0.5. The current randomization strategy aligns with the paper, but can be improved
            to avoid motion ambiguity. For example, one viable strategy is to only enable the mask dropout
            on a small portion of elements of the same body region.
    Returns:
        torch.Tensor: A tensor of shape (len(goal_state_names),) containing the mask.

    """
    # First we do the mode masking.
    mask_length = len(mask_element_names)
    mask = torch.zeros((num_envs, mask_length), dtype=torch.bool, device=device)

    modes = list(mask_modes.keys())
    # Pre-compute indices for all modes and patterns
    mode_to_indices = {}
    for mode in modes:
        all_indices = []
        for _, goal_state_patterns in mask_modes[mode].items():
            indices = get_matching_indices(goal_state_patterns, mask_element_names)
            all_indices.extend(indices)
        mode_to_indices[mode] = torch.tensor(sorted(set(all_indices)), dtype=torch.long, device=device)

    # Random select mode for each environment
    # Shape:1 x num_envs
    selected_mode_indices = torch.randint(0, len(modes), (num_envs,), device=device)

    for mode_idx, mode in enumerate(modes):
        # Using tensor operations for boolean selection
        mode_env_indices = (selected_mode_indices == mode_idx).nonzero(as_tuple=True)[0]
        if mode_env_indices.numel() > 0:
            indices = mode_to_indices[mode]
            mask[mode_env_indices.unsqueeze(1), indices] = True

    # Last we do the sparsity masking.
    if enable_sparsity_randomization:
        # Multiply by 0.5 to make it a probability.
        mask = torch.bernoulli(mask * 0.5).bool()

    return mask
