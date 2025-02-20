#!/bin/bash
#
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
set -e

run_unittests_in_folder() (
    # We use a subshell so we can cd into the folder.
    local folder=${1:?}
    cd ${folder}
    echo "Running unit tests in '${folder}'."
    ${ISAACLAB_PATH:?}/isaaclab.sh -p -m unittest
)

run_unittests_in_file() {
    local file=${1:?}
    echo "Running unit tests in '${file}'."
    ${ISAACLAB_PATH}/isaaclab.sh -p -m unittest "${file}"
}

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ${script_dir}

# Directories containing unit tests.
test_folders=(
    "."
    "neural_wbc/core"
    "neural_wbc/data"
    "neural_wbc/mujoco_wrapper"
    "neural_wbc/student_policy"
    "neural_wbc/inference_env"
)

# Run unit tests in each directory.
for folder in "${test_folders[@]}"; do
    run_unittests_in_folder "$folder"
done

# Additional files containing unit tests.
test_files=(
    "scripts/rsl_rl/tests/test_teacher_policy_cfg.py"
)

# Run unit tests in each file.
for file in "${test_files[@]}"; do
    run_unittests_in_file "$file"
done
