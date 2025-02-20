# Neural WBC - Student Policy

The `neural_wbc.policy` package provides tooling to train a student policy for neural whole body
controllers.

## Installation
To use the `neural_wbc.student_policy` package, install it with

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install -e neural_wbc/student_policy
```

## Usage
Refer to the Isaac Lab implementation of the neural WBC environment, as well as the training and
testing workflows, for examples of how to use the module.


## Unit Tests
The tests are located in the `tests` directory and can be run with the `unittest` module.

```bash
cd neural_wbc/student_policy
${ISAACLAB_PATH:?}/isaaclab.sh -p -m unittest
```
