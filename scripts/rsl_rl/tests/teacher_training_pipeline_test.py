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


import glob
import os
import subprocess
import tempfile

import neural_wbc.data
from neural_wbc.data import get_data_path


def main():
    motion_dataset_path = get_data_path("motions/stable_punch.pkl")
    policy_path, checkpoint = neural_wbc.data.get_default_teacher_policy()
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        command = (
            "${ISAACLAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train_teacher_policy.py "
            "--headless "
            "--num_envs=10 "
            "--teacher_policy.max_iterations=10 "
            "--teacher_policy.resume "
            f"--reference_motion_path={motion_dataset_path} "
            f"--teacher_policy.path={temp_dir} "
            f"--teacher_policy.resume_path={policy_path} "
            f"--teacher_policy.checkpoint={checkpoint} "
        )

        # Run the command
        subprocess.run(command, shell=True, check=True)

        # Find the results directory
        results_path_pattern = os.path.join(temp_dir, "*")
        results_directories = glob.glob(results_path_pattern)
        assert len(results_directories) == 1, "There should be exactly one results directory."
        results_dir_path = results_directories[0]

        # Verify the files exist in the results directory under the temporary directory
        expected_files = ["model_0.pt", "model_10.pt", "config.json"]
        for file in expected_files:
            file_path = os.path.join(results_dir_path, file)
            assert os.path.exists(file_path), f"{file} does NOT exist in the results directory."

        print("All expected files are present in the results directory.")


if __name__ == "__main__":
    main()
