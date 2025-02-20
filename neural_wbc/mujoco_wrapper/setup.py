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


from setuptools import find_packages, setup

setup(
    name="neural_wbc_mujoco_wrapper",
    version="0.1.0",
    packages=find_packages(),
    description="Mujoco wrapper for sim-to-sim validation",
    python_requires=">=3.10",
    install_requires=[
        "mujoco",
        "mujoco-python-viewer",
        "numpy",
        "torch",
        "tyro",
        "neural_wbc_core",
        "neural_wbc_data",
    ],
)
