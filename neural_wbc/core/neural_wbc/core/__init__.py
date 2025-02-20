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

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import sys
import types


def create_dummy_warp_module():
    """Creates a dummy 'warp' module with necessary attributes and submodules.

    This function is created because the 'warp' module can only be properly imported
    when Isaac Sim is also launched, which is not always required when running unit
    tests or simulations with other simulators. By creating a dummy module, we can
    avoid import errors and ensure that the dependent modules can be tested without
    requiring Isaac Sim to be running.
    """
    # Step 1: Create the main dummy module
    dummy_module = types.ModuleType("warp")

    # Step 2: Create the submodule 'torch'
    torch_submodule = types.ModuleType("torch")

    # Step 3: Define the array class
    class Array:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return f"Array(value={self.value})"

    # Define torch.to_torch and torch.from_torch functions
    def to_torch(value):
        return f"Converted {value} to torch"

    def from_torch(value):
        return f"Converted {value} from torch"

    # Step 4: Add the class to the main dummy module
    dummy_module.array = Array

    # Add the functions to the torch submodule
    torch_submodule.to_torch = to_torch
    torch_submodule.from_torch = from_torch

    # Add the torch submodule to the main dummy module
    dummy_module.torch = torch_submodule

    # Step 5: Insert the main dummy module into sys.modules
    sys.modules["warp"] = dummy_module


from .body_state import BodyState  # noqa
from .environment_wrapper import EnvironmentWrapper  # noqa
from .evaluator import Evaluator  # noqa
from .modes import NeuralWBCModes  # noqa
from .reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg, ReferenceMotionState  # noqa
from .termination import check_termination_conditions  # noqa
