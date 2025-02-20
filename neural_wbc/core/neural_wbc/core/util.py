# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# SPDX-License-Identifier: Apache-2.0
import re


def assert_equal(lhs: any, rhs: any, name: str):
    """Assert that 2 values are equal and provide a useful error if not."""
    assert lhs == rhs, f"{name}: Values are not equal: {lhs} != {rhs}"


def get_matching_indices(patterns: list[str], values: list[str], allow_empty: bool = False) -> list[int]:
    """Get indices of all elements in values that match any of the regex patterns."""
    all_indices = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        indices = [i for i, v in enumerate(values) if regex.match(v)]
        if len(indices) == 0 and not allow_empty:
            raise ValueError(f"No matching indices found for pattern {pattern} in {values}")
        all_indices.update(indices)
    return list(all_indices)
