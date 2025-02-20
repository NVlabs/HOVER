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

# We allow this to fail since in docker we only have the files without the git
# info.
git submodule update --init --recursive || true

echo "Resetting changes in third_party/human2humanoid..."
pushd third_party/human2humanoid
git reset --hard || true
popd

# Apply patch to files
if patch --dry-run --silent -f third_party/human2humanoid/phc/phc/utils/torch_utils.py < third_party/phc_torch_utils.patch; then
    echo "Dry run succeeded. Applying the patch..."
    patch third_party/human2humanoid/phc/phc/utils/torch_utils.py < third_party/phc_torch_utils.patch
    echo "Patch applied successfully."
else
    echo "Dry run failed. Patch was not applied. Check if the patch file contains errors."
    exit 1
fi

# Install libraries.
${ISAACLAB_PATH:?}/isaaclab.sh -p -m ensurepip
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install --upgrade pip
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install wheel
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install -e .
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install -r requirements.txt


# Check that IsaacLab is using the correct version.
if [ -d ${ISAACLAB_PATH}/.git ]; then
  expected_isaac_lab_tag="v2.0.0"
  if ! git -C ${ISAACLAB_PATH} tag -l "${expected_isaac_lab_tag}" | grep -q "${expected_isaac_lab_tag}"; then
      echo "Error: IsaacLab does not have this tag."
      echo "Expected tag: ${expected_isaac_lab_tag}"
      exit 1
  fi
fi
