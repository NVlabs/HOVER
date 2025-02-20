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


import os


def get_data_path(rel_path: str) -> str:
    """
    Get the absolute path to a data file located in the 'neural_wbc/data/data' directory.

    Args:
        rel_path (str): The relative path to the data file from the data directory.

    Returns:
        str: The absolute path to the data file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(file_dir, "../../data", rel_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file at '{rel_path}' does not exist in neural_wbc_data.")
    return data_path


def get_default_teacher_policy() -> tuple[str, str]:
    """Get the default teacher policy path and checkpoint."""
    return get_data_path("policy/h1:teacher"), "model_110000.pt"


def get_default_omnih2o_student_policy() -> tuple[str, str]:
    """Get the default omnih2o student policy path and checkpoint."""
    return get_data_path("policy/h1:student/omnih2o"), "model_7500.pt"


def get_default_hover_student_policy() -> tuple[str, str]:
    """Get the default hover student policy path and checkpoint."""
    return get_data_path("policy/h1:student/hover"), "model_51000.pt"
