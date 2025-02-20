# Neural WBC - Inference environment

High level inference environment used for evaluation and deployment.

## Installation

Install it with

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install -e neural_wbc/inference_env
```

## Usage

- First ensure that all dependencies are installed using the common script
    [./install_deps.sh](../../install_deps.sh)

- Run the simple viewer with:

    ```sh
    ${ISAACLAB_PATH}/isaaclab.sh -p neural_wbc/inference_env/scripts/mujoco_viewer_player.py
    ```

The viewer is paused on start by default, press `SPACE` key to start simulation or `RIGHT` arrow key
to step forward once. By default this script will load the UniTree H1 robot model scene from
[data/mujoco/models](../data/data/mujoco/models/scene.xml).

Additional scripts are provides for running comprehensive evaluation in [scripts](./scripts/)
directory

## Unit Tests

The tests are located in the `tests` directory and can be run with the `unittest` module.

```bash
cd neural_wbc/inference_env
${ISAACLAB_PATH:?}/isaaclab.sh -p -m unittest
```
