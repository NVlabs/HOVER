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


def to_numpy(x):
    """
    Check if input is a PyTorch tensor and convert to numpy array if true.
    Otherwise return the input unchanged.

    Args:
        x: Input to check and potentially convert

    Returns:
        numpy array if input was torch tensor, otherwise original input
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def squeeze_if_tensor(x, dim: int = 0):
    """
    Check if input is a PyTorch tensor and squeeze along the given dim if true.

    Args:
        x: Input to check and potentially convert
        dim: Dimension to squeeze

    Returns:
        numpy array if input was torch tensor, otherwise original input
    """
    if isinstance(x, torch.Tensor):
        return x.squeeze(dim=dim)
    return x
