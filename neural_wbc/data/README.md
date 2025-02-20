# Neural WBC - Data

The `neural_wbc.data` package provides utility functions to retrieve data files. It is designed to
help manage file paths in a structured and organized manner.

## Installation
To use the `neural_wbc.data` package, install it with

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install -e neural_wbc/data
```

## Usage
You can use the `get_data_path` function to retrieve the absolute path of a data file located in the
`neural_wbc/data/data` directory. The function raises a `FileNotFoundError` if the specified file
does not exist.

### Example
```python
from neural_wbc.data import get_data_path

try:
    data_file_path = get_data_path('mujoco/models/h1.xml')
    print(f"Data file path: {data_file_path}")
except FileNotFoundError as e:
    print(e)
```

## Data Files
The `data` directory contains the following subdirectories:
* `motions`: Contains motion datasets.
* `motion_lib`: Contains skeleton files used for loading motion datasets.
* `mujoco`: Contains files specific to the MuJoCo simulator, such as models and assets.
* `policy`: Contains policy checkpoints.


## Unit Tests
The tests are located in the `tests` directory and can be run with the `unittest` module.

```bash
cd neural_wbc/data
${ISAACLAB_PATH:?}/isaaclab.sh -p -m unittest
```
