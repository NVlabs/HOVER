#!/bin/bash
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

# Function to run a command and exit on failure
run_command() {
    local cmd="$1"
    echo "Running: $cmd"
    $cmd
    if [ $? -ne 0 ]; then
        echo "Command failed: $cmd"
        exit 1
    fi
}

# List of test scripts to run
test_scripts=(
    "neural_wbc/inference_env/tests/test_evaluation_pipeline.py"
    "scripts/rsl_rl/tests/student_evaluation_pipeline_test.py"
    "scripts/rsl_rl/tests/student_training_pipeline_test.py"
    "scripts/rsl_rl/tests/teacher_evaluation_pipeline_test.py"
    "scripts/rsl_rl/tests/teacher_training_pipeline_test.py"
)

# Execute each test script
for script in "${test_scripts[@]}"; do
    cmd="${ISAACLAB_PATH}/isaaclab.sh -p ${script}"
    run_command "$cmd"
done

echo "All commands executed successfully."
