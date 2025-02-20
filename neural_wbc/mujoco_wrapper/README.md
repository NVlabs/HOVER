# Neural WBC - Mujoco Wrapper

High level wrapper of Mujoco simulator for use in validating Whole Body Control (WBC) policies
trained in IsaacLab. The wrapper provides convenient interfaces for accessing data and manipulating
the underlying models in Mujoco, as well as an environment that allows comparison with IsaacLab.

```mermaid
graph LR

subgraph MJW[Mujoco Wrapper]
style MJW fill:#76B900,stroke:#333,stroke-width:2px
    A[Mujoco]
    style A fill:#CDCDCD,stroke:#333,stroke-width:2px
end

MJW -->|Data & Model| C[ Mujoco Visualizer]
MJW -->|Data| D[Evaluator]
style D fill:#76B900,stroke:#333,stroke-width:2px
style C fill:#CDCDCD,stroke:#333,stroke-width:2px

E[IsaacLab]
E[IsaacLab] -->|WBC Policy| MJW
style E fill:#76B900,stroke:#333,stroke-width:2px

RFM[Reference Motion Manager]
style RFM fill:#76B900,stroke:#333,stroke-width:2px

RFM -->| Reference Motions| MJW
```

## Installation

To use the `neural_wbc.mujoco_wrapper` package, install it with

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install -e neural_wbc/mujoco_wrapper
```

Also install the inference environment with

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

Additional scripts are provides for running comprehensive evaluation in [scripts](../inference_env/scripts/)
directory

**NOTE** The visualization of models is done using a modified version of
[mujoco-python-viewer](https://github.com/rohanpsingh/mujoco-python-viewer) with changes to allow
persistent markers for reference motion states. These modifications are bundled in
[third_party/mujoco_viewer](../../third_party/mujoco_viewer/) for self containment.

## Deployment

Once sim-to-sim validation is done, the policy can be deployed in simulation or on real robot.
This involves loading the relevant checkpoint and generating actions based on the observations
from the target robot. For real robots, where some quantities are not easily observable like
body orientations, use of a forward dynamics model is necessary.

## Unit Tests

The tests are located in the `tests` directory and can be run with the `unittest` module.

```bash
cd neural_wbc/mujoco_wrapper
${ISAACLAB_PATH:?}/isaaclab.sh -p -m unittest
```
