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


def convert_tensors_and_slices_to_serializable(d):
    """Recursively convert torch tensors to lists and handle slice objects in a nested dictionary, including lists and tuples."""
    if isinstance(d, dict):
        return {k: convert_tensors_and_slices_to_serializable(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return {"__type__": "tensor", "data": d.tolist()}
    elif isinstance(d, list):
        return [convert_tensors_and_slices_to_serializable(item) for item in d]
    elif isinstance(d, tuple):
        return {"__type__": "tuple", "data": tuple(convert_tensors_and_slices_to_serializable(item) for item in d)}
    elif isinstance(d, slice):
        return {"__type__": "slice", "start": d.start, "stop": d.stop, "step": d.step}
    else:
        return d


def convert_serializable_to_tensors_and_slices(d):
    """Recursively convert lists back to torch tensors and dictionaries back to slice objects."""
    if isinstance(d, dict):
        if "__type__" in d:
            if d["__type__"] == "tensor":
                return torch.tensor(d["data"])
            elif d["__type__"] == "slice":
                return slice(d["start"], d["stop"], d["step"])
            elif d["__type__"] == "tuple":
                return tuple(convert_serializable_to_tensors_and_slices(item) for item in d["data"])
        else:
            return {k: convert_serializable_to_tensors_and_slices(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_serializable_to_tensors_and_slices(item) for item in d]
    else:
        return d
