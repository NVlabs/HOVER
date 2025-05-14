# HOVER WBC

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/ubuntu-22.04-red)](https://releases.ubuntu.com/22.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-ApacheV2-yellow.svg)](https://opensource.org/license/apache-2-0/)

# Table of Contents

- [HOVER WBC](#hover-wbc)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Installation](#installation)
- [Training](#training)
  - [Data Processing](#data-processing)
    - [AMASS dataset](#amass-dataset)
  - [Teacher Policy](#teacher-policy)
  - [Student Policy](#student-policy)
  - [General Remarks for Training](#general-remarks-for-training)
  - [Generalist vs. Specialist Policy](#generalist-vs-specialist-policy)
- [Testing](#testing)
  - [Play Teacher Policy](#play-teacher-policy)
  - [Play Student Policy](#play-student-policy)
- [Evaluation](#evaluation)
- [Overwriting Configuration Values](#overwriting-configuration-values)
- [Sim-to-Sim Validation](#sim-to-sim-validation)
- [Developing](#developing)
  - [Unit testing](#unit-testing)
  - [Linting \& Formatting](#linting--formatting)
  - [Set up IDE (Optional)](#set-up-ide-optional)
  - [Running Scripts from an Isaac Lab Docker Container](#running-scripts-from-an-isaac-lab-docker-container)
- [License](#license)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

# Overview

This repository contains the IsaacLab extension to train neural whole-body controllers for humanoids
as explained in the [OmniH2O][omnih2o_paper] and [HOVER][hover_paper] papers. For video
demonstrations and to link to the original implementation in Isaac Gym, please visit the
[OmniH2O project website](https://omni.human2humanoid.com/) and the
[HOVER project website](https://hover-versatile-humanoid.github.io/).


<img src="docs/robots_in_action.gif" alt="Humanoid robots tracking motions from the AMASS dataset" width="400"/>

# Installation

1. Install Isaac Lab, see the [installation guide](https://isaac-sim.github.io/IsaacLab/v2.0.0/source/setup/installation/index.html).
    **Note**: Currently HOVER has been tested with Isaac Lab versions 2.0.0. After you clone the Isaac Lab
    ```bash
    git fetch origin
    git checkout v2.0.0
    ```
2. Define the following environment variable to specify the path to your IsaacLab installation:
    ```bash
    # Set the ISAACLAB_PATH environment variable to point to your IsaacLab installation directory
    export ISAACLAB_PATH=<your_isaac_lab_path>
    ```
3. Clone the repo and its submodules:
    ```bash
    git clone --recurse-submodules <REPO_URL>
    ```
4. Install this repo and its dependencies by running the following command from the root of this
   repo:
    ```bash
    pip install -r requirements.txt
    ```

# Training


## Data Processing

> **_NOTE:_** Due to the license limitations of the AMASS dataset, we are not able to provide the retargeted dataset directly.
All the following training and evaluation scripts will use the `stable_punch.pkl` dataset (not included as well)
as a toy example. It is a small subset of the AMASS dataset where the upper body is performing punching motions.
We modified the motion data to minimize the lower body's motion to create a simpler example. We suggest that users retarget a small subset of the AMASS dataset to the
Unitree H1 robot and use that for trial training. The retargeting process of the whole dataset could take up to
4 days on a 32 CPU core machine. More cores will reduce the time correspondingly.

### AMASS dataset

We utilize the AMASS dataset to train our models. The AMASS dataset is a comprehensive collection of
motion capture (mocap) datasets. To develop control policies for a humanoid robot, it is essential
to retarget the motion data in the dataset to fit the desired robot. We provide a bash script that
retargets the dataset specifically for the Unitree H1 robot. This script is based on the scripts
from the [human2humanoid repository](https://github.com/LeCAR-Lab/human2humanoid). Due to the
limitations of the [license](https://amass.is.tue.mpg.de/license.html) of the AMASS dataset, we are
not providing a retargeted dataset directly. To access the dataset, you will need to create an account.

To get started, follow these steps:

1. Create a folder to save the datasets in `mkdir -p third_party/human2humanoid/data/AMASS/AMASS_Complete`.
2. Download the dataset(s) you are interested in from the "SMPL+H G" format section on the [AMASS download
   page](https://amass.is.tue.mpg.de/download.php) and place the archive files in
   `third_party/human2humanoid/data/AMASS/AMASS_Complete`. This will take some time due to the number of
   datasets and the fact that apparently they don't allow parallel downloads. You don't need to extract
   the files manually - the script will handle that for you.
3. Download the SMPL model from [this
   link](https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip) and
   place the zip file `third_party/human2humanoid/data/smpl`.
4. Finally, run the provided script by executing `./retarget_h1.sh`. The script extracts the
   downloaded files to desired locations, prepares necessary files and dependencies for retargeting.
   If you want to retarget only specific motions, you can provide a YAML file with the list of motions by running `./retarget_h1.sh --motions-file <path_to_yaml_file>`. See `punch.yaml` for an example. This will only process the motion files specified in the YAML file instead of the full dataset.
   Note that the script installs pip dependencies and might build some of them, which requires the matching
   version of the `python-dev` to be installed.
5. Before proceeding with training and evaluation, run `./install_deps.sh` again to ensure the
  correct dependencies are installed.


The retargeted dataset will be found at `third_party/human2humanoid/data/h1/amass_all.pkl`. Rename it and move it to your desired location.
While the exact path to the reference motion is not important, we recommend placing it in the `neural_wbc/data/data/motions/`
folder as the included data library will handle relative path searching, which is useful for unit testing.

For more details, refer to the [human2humanoid repository](https://github.com/LeCAR-Lab/human2humanoid/tree/main?tab=readme-ov-file#motion-retargeting).

## Teacher Policy


In the project's root directory,

```bash
python scripts/rsl_rl/train_teacher_policy.py \
    --num_envs 1024 \
    --reference_motion_path neural_wbc/data/data/motions/stable_punch.pkl # you can omit this as it's the default
```

The teacher policy is trained for 10000000 iterations, or until the user interrupts the training. The resulting checkpoint is stored in `<data_time>` and the filename is `model_<iteration_number>.pt`.


## Student Policy


In the project's root directory,
```bash
python scripts/rsl_rl/train_student_policy.py \
    --num_envs 1024 \
    --headless
```
This assumes that you have already trained the teacher policy as there is no provided teacher policy in the repo. Change the filename to match the checkpoint you trained.
The exact path of the teacher policy does not matter, but it is recommended to store it in the data folder. If stored outside the data folder, you might need to provide the full path.

## Multi-GPU
It is possible to train on a node with multiple gpus or multiple nodes with multiple gpues.
In order to train the teacher on a machine with 4 gpus:
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 scripts/rsl_rl/train_teacher.py --num_envs=1024 --headless --distributed
```
This will generate 4 IsaacSim simulation, each containing 1024 environments.

## General Remarks for Training

- The examples above use a low number of environments as a toy demo. For good results we
    recommend to train with at least 4096 environments.
- The examples above use the `stable_punch.pkl` dataset as a toy demo. For good
    results we recommend to train with the full amass dataset.
- Per default the trained checkpoints are stored to `logs/teacher/` or `logs/student/`.
- If you don't want to train from scratch you can resume training from a checkpoint using the
    options `--load_run=<date_time>` and/or `--checkpoint=<model_iter.pt>`. For example to resume training of
    the teacher use

    ```bash
    python scripts/rsl_rl/train_teacher_policy.py \
        --num_envs 10 \
        --load_run 2025-05-09_09-23-50 \
        --checkpoint model_500.pt

    ```

- Training requires a single GPU, we found the following performance when training on different
    GPUs:
    <br>

    **Teacher Training:**
    | GPU       | Num Iterations | Time per Iteration (s) | Training Time (h) |
    |-----------|----------------|------------------------|-------------------|
    | RTX 4090  | 100'000        | 0.84                   | 23.3              |
    | RTX A6000 | 100'000        | 1.90                   | 52.8              |
    | L40       | 100'000        | 1.61                   | 44.6              |

    **Student Training:**
    | GPU       | Num Iterations | Time per Iteration (s) | Training Time (h) |
    |-----------|----------------|------------------------|-------------------|
    | RTX 4090  | 10'000         | 0.097                  | 0.27              |
    | RTX A6000 | 10'000         | 0.18                   | 0.50              |
    | L40       | 10'000         | 0.176                  | 0.49              |

## Generalist vs. Specialist Policy

The codebase allows to train both generalist and specialist policies:
- Generalist policies allow to track different command configurations (or modes) with a single policy, as shown
    in the [HOVER][hover_paper] paper.
- Specialist policies only allow to track a specific command configuration with a single policy, as
    shown in the [OmniH2O][omnih2o_paper] paper.

Per default the codebase trains a specialist policy in OmniH2O mode (tracking head and hand positions).
```py
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
```
A specialist in a different mode can be trained by modifying the `distill_mask_modes` in
the [config file](./neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/neural_wbc_env_cfg_h1.py#73).
For an example to train a specialist that tracks the joint angles, root linear velocity and root yaw
orientation use this:
```py
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {"humanplus": DISTILL_MASK_MODES_ALL["humanplus"]}
```
A generalist can be trained by removing/commenting out the specialist mask modes in the
[config file](./neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/neural_wbc_env_cfg_h1.py#73), ie.

```py
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = DISTILL_MASK_MODES_ALL
```

In the current implementation, we hand picked four modes that are discussed in the original paper
for proof of life purposes. The user is free to add more modes to the `DISTILL_MASK_MODES_ALL`
dictionary to make the generalist policy more general. We recommend the user to turn off sparsity
randomization as the currently implemented randomization strategy (as described in the paper) might
lead to motion ambiguity.

In both cases the same commands from above can be used to launch the training.


# Testing

## Play Teacher Policy

In the project's root directory,

```bash
python scripts/rsl_rl/play.py \
    --num_envs 10 \
    --reference_motion_path neural_wbc/data/data/motions/stable_punch.pkl \
```
This will run the latest checkpoint in the latest run.
Alternatively you can specify the run from which you want to load the checkpoint from by passing `--load_run=logs/rsl_rl/teacher_policy/h1/<date_time>`, which will use the latest checkpoint in the folder. You can also specify a specific checkpoint by passing `--checkpoint=<model_iter.pt>`

## Play Student Policy

In the project's root directory,

```bash
python scripts/rsl_rl/play.py \
    --num_envs 10 \
    --student_player \
```

# Evaluation

The evaluation iterates through all the reference motions included in the dataset specified by the
`--reference_motion_path` option and exits when all motions are evaluated. Randomization is turned
off during evaluation. At the end of execution, the script summarizes the results with the following
reference motion tracking metrics:

> **_NOTE:_** Running evaluation with 1024 environments will require approximately 13GB of GPU memory. Adjust the `--num_envs` parameter based on your available GPU resources.

* **Success Rate [%]**: The percentage of motion tracking episodes that are successfully completed. An
    episode is considered successful if it follows the reference motion from start to finish without
    losing balance and avoiding collisions on specific body parts.
* **mpjpe_g [mm]**: The global mean per-joint position error, which measures the policy's ability to
    imitate the reference motion globally.
* **mpjpe_l [mm]**: The root-relative mean per-joint position error, which measures the policy's ability
    to imitate the reference motion locally.
* **mpjpe_pa [mm]**: The procrustes aligned mean per-joint position error, which aligns the links with
    the ground truth before calculating the errors.
* **accel_dist [mm/frame^2]**: The average joint acceleration error.
* **vel_dist [mm/frame]**: The average joint velocity error.
* **upper_body_joints_dist [radians]**: The average distance between the predicted and ground truth upper body joint positions.
* **lower_body_joints_dist [radians]**: The average distance between the predicted and ground truth lower body joint positions.
* **root_r_error [radians]**: The average torso roll error.
* **root_p_error [radians]**: The average torso pitch error.
* **root_y_error [radians]**: The average torso yaw error.
* **root_vel_error [m/frame]**: The average torso velocity error.
* **root_height_error [m]**: The average torso height error.

The metrics are reported multiple times for different configurations:
    - The metrics are computed over all environments or only the successful ones.
    - The metrics are computed over all bodies or only the tracked (=masked) bodies.

Per default the masked evaluation is using the OmniH2O mode (tracking head and hand positions). This
can be configured by changing the mask here the `distill_mask_modes` in the
[config file](./neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/neural_wbc_env_cfg_h1.py#73).
For an example to change the configuration see also the
[Generalist vs. Specialist Policy section](#generalist-vs-specialist-policy) above.


The evaluation script, `scripts/rsl_rl/eval.py`, uses the same arguments as the play script,
`scripts/rsl_rl/play.py`. You can use it for both teacher and student policies.

```bash
# For a teacher
python scripts/rsl_rl/eval.py \
    --num_envs 10 \
```

``` bash
# For a student
python scripts/rsl_rl/eval.py \
    --num_envs 10 \
    --student_player
```
# Overwriting Configuration Values

To customize and overwrite default environment configuration values, you can provide a YAML file
with the desired settings. The structure of the YAML file should reflect the hierarchical structure
of the configuration. For nested configuration parameters, use a dot (.) to separate the levels. For
instance, to update the `dt` value within the sim configuration, you would use `sim.dt` as the key.
Here's an example YAML file demonstrating how to set and overwrite various configuration values:

```yaml
# scripts/rsl_rl/config_overwrites/sample.overwrite.yaml
sim.dt: 0.017
decimation: 4
add_policy_obs_noise: False
default_rfi_limit: 0.1
ctrl_delay_step_range: [0, 3]
```

To apply these custom settings, pass the path to your YAML file using the `--env_config_overwrite`
option when running the script. If the YAML file contains keys that do not exist in the default
configuration, those keys will be ignored.

# Validation
The trained policy in Isaac Lab can be validated in two ways - Sim-to-Sim and Sim-to-Real.

<div align="center">
<table>
  <tr>
    <td><img src="docs/stable_punch.gif" width="400"/></td>
    <td><img src="docs/stable_wave.gif" width="385"/></td>
  </tr>
  <tr>
    <td align="center"><font size="1"><em>Stable Punch - Mujoco (left) & Real Robot (right)</em></font></td>
    <td align="center"><font size="1"><em>Stable Wave - Mujoco (left) & Real Robot (right)</em></font></td>
  </tr>
</table>
</div>

## Sim-to-Sim Validation

We also provide a [Mujoco environment](./neural_wbc/mujoco_wrapper/) for conducting sim-to-sim validation of the trained policy.
To run the evaluation of Sim2Sim,

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p neural_wbc/inference_env/scripts/eval.py \
    --num_envs 1 \
    --headless \
    --student_path neural_wbc/data/data/policy/h1:student/ \
    --student_checkpoint model_<iteration_number>.pt
```

Please be aware that the `mujoco_wrapper` only supports one environment at a time. For a reference, it will take up to `5h` to evaluate `8k` reference motions. The inference_env is designed for maximum versatility.

## Sim-to-Real Deployment

For sim-to-real validation, we provide a [Hardware environment](./neural_wbc/hw_wrappers/) for Unitree H1.
To install the required dependencies and environment setup, please refer to the [README of sim2real deployment](./neural_wbc/hw_wrappers/README.md).

To deploy the trained policy on the [Unitree H1 robot](https://unitree.com/h1),

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p neural_wbc/inference_env/scripts/s2r_player.py \
    --student_path neural_wbc/data/data/policy/h1:student/ \
    --student_checkpoint model_<iteration_number>.pt \
    --reference_motion_path neural_wbc/data/data/motions/<motion_name>.pkl \
    --robot unitree_h1 \
    --max_iterations 5000 \
    --num_envs 1
```

> **_NOTE:_** The sim-to-real deployment wrapper currently only supports the Unitree H1 robot. It can be extended to other robots by implementing the corresponding hardware wrapper interface.

# Developing

## Unit testing

In the root of each module directory (e.g. `neural_wbc/core`), run the following command:

```bash
cd neural_wbc/core
${ISAACLAB_PATH:?}/isaaclab.sh -p -m unittest
```
We do provide a `run_unit_tests.sh` script to run all unit tests.

## Linting & Formatting

We have a pre-commit template to automatically lint and format the code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the
    `setup_python_env` in the drop down menu. When running this task, you will be prompted to add
    the absolute path to your Isaac Lab installation.

If everything executes correctly, it should create a file .python.env in the .vscode directory. The
file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps
in indexing all the python modules for intelligent suggestions while writing code.


## Running Scripts from an Isaac Lab Docker Container

You can run scripts in a Docker container without using the Isaac Sim GUI. Follow these steps:

1. **Install the NVIDIA Container Toolkit**:
   - Follow the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

2. **Access the NGC Container Registry**:
   - Ensure you have access by following the instructions [here](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#accessing-ngc-registry).

3. **Start the Docker Container**:
   - Use the following command to start the container:

    ```bash
     docker run -it --rm \
         --runtime=nvidia --gpus all \
         -v $PWD:/workspace/neural_wbc \
         --entrypoint /bin/bash \
         --name neural_wbc \
         nvcr.io/nvidian/isaac-lab:IsaacLab-main-b120
    ```

4. **Set Up the Container**:
   - Navigate to the workspace and install dependencies:

     ```bash
     cd /workspace/neural_wbc
     ./install_deps.sh
     ```

You can now run scripts in headless mode by passing the `--headless` option.


# License

HOVER WBC is released under the Apache License 2.0. See LICENSE for additional details.

# Contributors

> The names are ordered in alphabetical order by the last name:

Joydeep Biswas, Yan Chang, Jim Fan, Pulkit Goyal, Lionel Gulich, Tairan He, Rushane Hua, Neel Jawale, H. Hawkeye King, Chenran Li, Michael Lin, Wei Liu, Zhengyi Luo, Billy Okal, Stephan Pleines, Soha Pouya, Peter Varvak, Wenli Xiao, Huihua Zhao, Yuke Zhu

# Acknowledgments

We would like to acknowledge the following projects where parts of the codes in this repo is derived from:

- [Mujoco Python Viewer](https://github.com/rohanpsingh/mujoco-python-viewer)
- [RSL RL](https://github.com/leggedrobotics/rsl_rl)
- [human2humanoid](https://github.com/LeCAR-Lab/human2humanoid)
- [Unitree Python SDK](https://github.com/unitreerobotics/unitree_sdk2_python)

[omnih2o_paper]: https://arxiv.org/abs/2406.08858
[hover_paper]: https://arxiv.org/abs/2410.21229
