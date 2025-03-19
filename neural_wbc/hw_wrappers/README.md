# Neural WBC - Hardware Wrapper for Unitree H1

This module is a high-level wrapper for the [Unitree H1](https://unitree.com/h1) robot hardware. It is designed to provide convenient interfaces to deploy the trained policy on the Unitree H1 robot. The main purpose of this wrapper is to validate Whole Body Control (WBC) policies trained in IsaacLab on the real robot.
> **Note:**  The sim-to-real deployment wrapper currently only supports the Unitree H1 robot. It can be extended to other robots by implementing the corresponding hardware wrapper interface.

> **Note:** This guide assumes familiarity with basic Unitree H1 operation. Please refer to the official [Unitree H1 Operation Guide](https://support.unitree.com/home/en/H1_developer/quick_start) for fundamental robot operations, as these are outside the scope of this guide.

## Setup
Our deployment setup uses an Ubuntu PC with GPU for running policy inference, connected to the H1 robot via Ethernet. For network configuration, please refer to the [Unitree H1 Quick Start Guide](https://support.unitree.com/home/en/H1_developer/start). The IMU on-board the H1 robot is used for estimating the rotational velocity and gravity vector required as input to the policy.

<div style="display: flex; justify-content: center;">

| Component    | Specification              |
|:----------------|:------------------:|
| Control loop Rate    | 200 Hz              |
| Policy inference rate    | 50 Hz             |
| Robot Type      | Unitree H1         |
| Environment   | Linux |
| GPU   | NVIDIA GeForce RTX 4090 |
</div>


> **Note:** While these specifications detail our deployment environment, the policy can be deployed using any computer with similar capabilities. Performance may vary based on your specific hardware configuration. The student policy is light enough to be inferenced on a CPU as well.

<div align="center">
<div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; display: inline-block;">
<img src="docs/unitree_h1_setup.png" width="400"/>
<br>
<font size="1"><em>Unitree H1 setup</em></font>
</div>
</div>

## Installation

To install the required dependencies:

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install -r requirements_deploy.txt
```
For any issues with Unitree SDK installation, see the [Unitree Python SDK](https://github.com/unitreerobotics/unitree_sdk2_python).

## Usage

### Check for successful connection with the robot
- Make sure the robot is in develop mode using the joystick before executing the below example. Use the [Unitree Operation guide](https://support.unitree.com/home/en/H1_developer/quick_start) to transition the robot to develop mode.
- Make sure robot is safely mounted on gantry with enough space around it.
    ```bash
    cd ~/unitree_sdk2_python
    python3 example/h1/low_level/h1_low_level_example.py <network interface>
    ```
If the robot responds to the low-level example commands, your setup and network connection are working correctly. You can proceed to the next steps after exiting the example program.

### Check for successful setup of the Hover stack:
- Execute the below commands from the hover stack root directory.
- First ensure that all dependencies are installed using the common script
    [./install_deps.sh](../../install_deps.sh)

- Run the simple viewer with:

    ```sh
    ${ISAACLAB_PATH}/isaaclab.sh -p neural_wbc/inference_env/scripts/mujoco_viewer_player.py
    ```

The viewer is paused on start by default, press `SPACE` key to start simulation or `RIGHT` arrow key
to step forward once. By default this script will load the UniTree H1 robot model scene from
[data/mujoco/models](../data/data/mujoco/models/scene.xml).

## Deployment

> **Note:** Current deployment does not use external sensors for robot root tracking. The setup supports deploying stable motions where the robot stabilizes in place while executing upper body motions.

By default the policy is deployed in OmniH2O mode (tracking head and hand positions). In order to deploy a different mode, change the configuration in [config_file](../inference_env/inference_env/neural_wbc_env_cfg_real_h1.py).

Once sim-to-sim validation is done, the policy can be deployed on the real robot using the following steps.

1. Follow the [Unitree H1 Operation Guide](https://support.unitree.com/home/en/H1_developer/quick_start) to:
    - Boot the robot (Steps 1-5 under "Boot Process")
    - Switch to develop mode using the joystick (Steps under "Develop Mode")
2. Once the H1 robot is transitioned to develop mode, execute this command to deploy the policy while the robot is securely mounted on the gantry:
    ```bash
    ${ISAACLAB_PATH:?}/isaaclab.sh -p neural_wbc/inference_env/scripts/s2r_player.py \
        --student_path neural_wbc/data/data/policy/h1:student/ \
        --student_checkpoint model_<iteration_number>.pt \
        --reference_motion_path neural_wbc/data/data/motions/<motion_name>.pkl \
        --robot unitree_h1 \
        --max_iterations 5000 \
        --num_envs 1
    ```
    You can now deploy the policy on the real robot in headless mode by passing the `--headless` option.

2. Upon command execution, the robot will automatically move to match the starting pose of the reference motion. You can preview this motion in the Mujoco viewer UI, where the reference trajectory is visualized as red dots.

3. Once the robot is in the starting pose:
   - Gradually lower the gantry until the robot's feet make contact with the ground
   - Press the space bar in the Mujoco viewer UI to begin policy execution
   - Completely lower the gantry to allow free movement of the robot

<div align="center">
<img src="docs/push_recovery.gif" width="400"/>
<br>
<font size="1"><em> Policy deployment of stable motion</em></font>
</div>

> **Note:** If the robot loses motor torque and becomes unresponsive to joystick commands, restart the robot and contact Unitree support for troubleshooting.

## Unit Tests

The tests are located in the `tests` directory and can be run with the `unittest` module.

```bash
cd neural_wbc/hw_wrappers
${ISAACLAB_PATH:?}/isaaclab.sh -p -m unittest
```
