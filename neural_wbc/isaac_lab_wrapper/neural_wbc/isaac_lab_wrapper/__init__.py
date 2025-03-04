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


import gymnasium as gym

from . import neural_wbc_env_cfg_h1

##
# Register Gym environments.
##

gym.register(
    id="NeuralWBC-H1-v0",
    entry_point="neural_wbc.isaac_lab_wrapper.neural_wbc_env:NeuralWBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": neural_wbc_env_cfg_h1.NeuralWBCEnvCfgH1,
    },
)
