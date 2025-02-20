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


import json
import os
import pprint
import subprocess
import sys
import tempfile

import neural_wbc.data
from neural_wbc.core import util


def get_json_files(directory):
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            json_files.append(os.path.join(directory, filename))
    return json_files


def compare_generated_metrics(generated_metrics):
    # Make sure we always have 100% success rate with the stable_punch dataset and default model
    util.assert_equal(generated_metrics["success_rate"], 1.0, "Success rate")
    # Make sure all and success metrics are the same
    util.assert_equal(generated_metrics["all"], generated_metrics["success"], "All metrics vs Success metrics")
    # Make sure we played the correct number of reference motions
    util.assert_equal(generated_metrics["num_motions"], 29, "Number of motions")


if __name__ == "__main__":
    metrics_path = tempfile.TemporaryDirectory()
    policy_path, checkpoint = neural_wbc.data.get_default_omnih2o_student_policy()
    eval_pipeline_command = [
        "${ISAACLAB_PATH}/isaaclab.sh -p neural_wbc/inference_env/scripts/eval.py "
        "--headless "
        f"--student_path={policy_path} "
        f"--student_checkpoint={checkpoint} "
        f"--metrics_path={metrics_path.name} "
    ]
    return_value = subprocess.call(eval_pipeline_command, shell=True)
    json_files = get_json_files(metrics_path.name)
    assert len(json_files) == 1
    with open(json_files[0], encoding="utf-8") as fh:
        generated_metrics = json.load(fh)
    print("Generated metrics:")
    pprint.pprint(generated_metrics)
    compare_generated_metrics(generated_metrics)
    sys.exit(return_value)
