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


import contextlib
import json
import os
import tempfile
import torch
import unittest
from dataclasses import _MISSING_TYPE, fields, is_dataclass

from neural_wbc.isaac_lab_wrapper.utils import convert_serializable_to_tensors_and_slices

from .test_main import APP_IS_READY

if APP_IS_READY:
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1


def deep_compare_dicts(dict1, dict2):
    """Recursively compare two dictionaries, including torch tensors."""
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]
        if isinstance(value1, dict) and isinstance(value2, dict):
            if not deep_compare_dicts(value1, value2):
                return False
        elif isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            if not torch.equal(value1, value2):
                return False
        elif (isinstance(value1, list) and isinstance(value2, list)) or (
            isinstance(value1, tuple) and isinstance(value2, tuple)
        ):
            if len(value1) != len(value2):
                return False
            for item1, item2 in zip(value1, value2):
                if not deep_compare_dicts({"item": item1}, {"item": item2}):
                    return False
        elif isinstance(value1, slice) and isinstance(value2, slice):
            if (value1.start, value1.stop, value1.step) != (value2.start, value2.stop, value2.step):
                return False
        elif value1 != value2:
            return False
    return True


def compare_configs(config1, config2):
    if not isinstance(config1, type(config2)):
        return False

    for field in fields(config1):
        attr_name = field.name
        attr_value = getattr(config1, attr_name)
        other_attr_value = getattr(config2, attr_name)
        if isinstance(attr_value, torch.Tensor) and isinstance(other_attr_value, torch.Tensor):
            if not torch.equal(attr_value, other_attr_value):
                return False
        elif is_dataclass(attr_value):
            if not compare_configs(attr_value, other_attr_value):
                return False
        elif isinstance(attr_value, dict):
            if not deep_compare_dicts(attr_value, other_attr_value):
                return False
        elif isinstance(attr_value, _MISSING_TYPE) and isinstance(other_attr_value, _MISSING_TYPE):
            pass
        elif attr_value != other_attr_value:
            return False

    return True


class TestNeuralWBCEnvCfgH1(unittest.TestCase):
    def setUp(self):
        temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".json")
        self.temp_file_name = temp_file.name
        temp_file.close()  # Close the file to prevent creation

        self.env_cfg = NeuralWBCEnvCfgH1()
        self.env_cfg.save(self.temp_file_name)

    def tearDown(self):
        with contextlib.suppress(OSError, FileNotFoundError):
            os.remove(self.temp_file_name)

    def test_save_config(self):
        with open(self.temp_file_name) as fh:
            loaded_cfg = json.load(fh)
        self.assertTrue(
            deep_compare_dicts(self.env_cfg.to_dict(), convert_serializable_to_tensors_and_slices(loaded_cfg))
        )

    def test_load_config(self):
        loaded_cfg = NeuralWBCEnvCfgH1()
        loaded_cfg.load(self.temp_file_name)
        self.assertTrue(compare_configs(self.env_cfg, loaded_cfg))
