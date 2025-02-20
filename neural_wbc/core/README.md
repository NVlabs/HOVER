# Neural WBC - Core

The `neural_wbc.core` package provides core functionalities for implementing neural whole-body
control for humanoid robots. It includes

* **Data Structures**
  - `BodyState` for managing robot body states.
  - `ReferenceMotionState` for managing motion states from a dataset.
* **Algorithm Functions**: Key functions for observations and termination conditions.
* **Motion Dataset Reader**: `ReferenceMotionManager` for reading and querying motion datasets.
* **Environment Wrapper**: Ensures compatibility with different simulators.
* **Evaluator**: Facilitates easy evaluation of trained policies.

## Installation
To use the `neural_wbc.core` package, install it with

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install -e neural_wbc/core
```

## Usage
Refer to the Isaac Lab implementation of the OmniH2O environment, as well as the training and
testing workflows, for examples of how to use the module.

## Evaluation and Metrics
The `Evaluator` class is used to collect data and evaluate the motion tracking performance of a
reinforcement learning (RL) policy in a simulated environment. Example usage:

```python
from neural_wbc.core import Evaluator
from your_module import YourEnvironmentWrapper

# Step 1: Initialization
env_wrapper = YourEnvironmentWrapper()  # Assume this is correctly set up
metrics_path = "path/to/save/metrics"  # Optional
evaluator = Evaluator(env_wrapper=env_wrapper, metrics_path=metrics_path)

# Simulate environment loop
while not evaluator.is_evaluation_complete():
    # Get observations, rewards, dones, and extra info from the environment
    actions = policy(observations)
    observations, rewards, dones, extras = env_wrapper.step(actions)

    # Step 2: Collect data from each step
    reset_env = evaluator.collect(dones=dones, info=extras)

    # Step 3: Reset environment if needed
    if reset_env:
        evaluator.forward_motion_samples()
        observations, _ = env_wrapper.reset()

# Step 4: Concluding the evaluation
evaluator.conclude()
```

The evaluator provides the following metrics:

* **Success Rate [%]**: The percentage of motion tracking episodes that are successfully completed. An
    episode is considered successful if it follows the reference motion from start to finish without
    losing balance and avoiding collisions on specific body parts.
* **mpjpe_g [mm]**: The global mean per-joint position error, which measures the policy’s ability to
    imitate the reference motion globally.
* **mpjpe_l [mm]**: The root-relative mean per-joint position error, which measures the policy’s ability
    to imitate the reference motion locally.
* **mpjpe_pa [mm]**: The procrustes aligned mean per-joint position error, which aligns the links with
    the ground truth before calculating the errors.
* **accel_dist [mm/frame^2]**: The average joint acceleration error.
* **vel_dist [mm/frame]**: The average joint velocity error.


## Unit Tests
The tests are located in the `tests` directory and can be run with the `unittest` module.

### Running the Tests
To run the unit tests, you can use the `unittest` module.

```bash
cd neural_wbc/core
${ISAACLAB_PATH:?}/isaaclab.sh -p -m unittest
```
