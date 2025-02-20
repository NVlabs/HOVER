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

HUMAN2HUMANOID_DIR="third_party/human2humanoid"
AMASS_DIR="$HUMAN2HUMANOID_DIR/data/AMASS/AMASS_Complete"
SMPL_DIR="$HUMAN2HUMANOID_DIR/data/smpl"
SMPL_MODEL_DIR="$SMPL_DIR/SMPL_python_v.1.1.0/smpl/models"

# Function to check and extract AMASS dataset
check_amass() {
    if [ -d "$AMASS_DIR" ]; then
        if find "$AMASS_DIR" -type f -name "*.npz" | grep -q . ; then
            echo "AMASS dataset is already extracted and ready."
        elif compgen -G "$AMASS_DIR/*.tar.bz2" > /dev/null || compgen -G "$AMASS_DIR/*.zip" > /dev/null; then
            echo "Extracting compressed files..."
            # Extract .tar.bz2 files
            find "$AMASS_DIR" -name "*.tar.bz2" -exec tar -xvjf {} -C "$AMASS_DIR" \;
            # Extract .zip files
            find "$AMASS_DIR" -name "*.zip" -exec unzip -o {} -d "$AMASS_DIR" \;
        else
            echo "Please download the AMASS dataset in the 'SMPL + H G' format from https://amass.is.tue.mpg.de/index.html and place it under $AMASS_DIR"
            exit 1
        fi
    else
        echo "$AMASS_DIR folder does not exist. Please create it and download the AMASS dataset in the 'SMPL + H G' format from https://amass.is.tue.mpg.de/index.html there."
        exit 1
    fi
}

check_files_exist() {
    for file in "$@"; do
        if [ ! -f "$file" ]; then
            return 1
        fi
    done
    return 0
}

# Function to check and extract SMPL files
check_smpl() {
    FEMALE_MODEL="$SMPL_MODEL_DIR/basicmodel_f_lbs_10_207_0_v1.1.0.pkl"
    MALE_MODEL="$SMPL_MODEL_DIR/basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
    NEUTRAL_MODEL="$SMPL_MODEL_DIR/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    MODELS=($FEMALE_MODEL $MALE_MODEL $NEUTRAL_MODEL)

    RENAMED_FEMALE_MODEL="$SMPL_DIR/SMPL_FEMALE.pkl"
    RENAMED_MALE_MODEL="$SMPL_DIR/SMPL_MALE.pkl"
    RENAMED_NEUTRAL_MODEL="$SMPL_DIR/SMPL_NEUTRAL.pkl"
    RENAMED_MODELS=($RENAMED_FEMALE_MODEL $RENAMED_MALE_MODEL $RENAMED_NEUTRAL_MODEL)

    if [ -d "$SMPL_DIR" ]; then
        if check_files_exist $RENAMED_MODELS[@]; then
            echo "SMPL files are already available."
        else
            if ! check_files_exist $MODELS[@]; then
                if [ ! -f "$SMPL_DIR/SMPL_python_v.1.1.0.zip" ]; then
                    echo "Please download the SMPL files from https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip and place it under $SMPL_DIR."
                    exit 1
                fi
                echo "Extracting SMPL_python_v.1.1.0.zip..."
                # -o to overwrite any existing files.
                unzip "$SMPL_DIR/SMPL_python_v.1.1.0.zip" -o -d "$SMPL_DIR"
            fi

            echo "Renaming SMPL model files..."
            cp $FEMALE_MODEL $RENAMED_FEMALE_MODEL
            cp $MALE_MODEL $RENAMED_MALE_MODEL
            cp $NEUTRAL_MODEL $RENAMED_NEUTRAL_MODEL
        fi
    else
        echo "$SMPL_DIR folder does not exist. Please create it and download the SMPL files from https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip there."
        exit 1
    fi
}

apply_patch() {
    local target_file=$1
    local patch_file=$2

    echo "Applying $patch_file to $target_file"
    if patch --dry-run --quiet -f "$target_file" < "$patch_file"; then
        echo "Dry run succeeded. Applying the patch..."
        patch "$target_file" < "$patch_file"
        echo "Patch applied successfully."
    fi
}

retarget() {
    # Install required packages to run the retargeting scripts.
    echo "Installing Python requirements..."
    pip install -r requirements.txt || return 1

    # Fit the SMPL shape that matches the H1 kinematics
    echo "Running grad_fit_h1_shape.py..."
    python scripts/data_process/grad_fit_h1_shape.py || return 1

    # Retarget the AMASS dataset based on the corresponding keypoints between fitted SMLP shape and H1
    echo "Running grad_fit_h1.py..."
    python scripts/data_process/grad_fit_h1.py --amass_root data/AMASS/AMASS_Complete || return 1

    return 0
}

# Check AMASS dataset
check_amass

# Check SMPL files
check_smpl

# Apply patches
apply_patch $HUMAN2HUMANOID_DIR/requirements.txt third_party/requirements.patch || exit 1
apply_patch $HUMAN2HUMANOID_DIR/scripts/data_process/grad_fit_h1.py third_party/grad_fit_h1.patch || exit 1

echo "Move to $HUMAN2HUMANOID_DIR"
pushd $HUMAN2HUMANOID_DIR

retarget
if [ $? -ne 0 ]; then
    echo "Motion retargeting failed."
else
    echo "Motion retargeting finished. You should find the retargeted dataset at $HUMAN2HUMANOID_DIR/data/h1/amass_all.pkl."
fi

# Change back to the original directory
popd
