---
sidebar_label: preprocessing
title: preprocessing.preprocessing
---

#### logger\_config

#### logger

#### preprocessing

```python
@save_state(model=PreprocessingFunctionSchema)
def preprocessing(config: dict,
                  centered_reference_keypoint: str,
                  orientation_reference_keypoint: str,
                  run_lowconf_cleaning: bool = True,
                  run_egocentric_alignment: bool = True,
                  run_outlier_cleaning: bool = True,
                  run_savgol_filtering: bool = True,
                  run_rescaling: bool = True,
                  save_logs: bool = True) -> str
```

Preprocess the data by:
    - Cleaning low confidence data points
    - Egocentric alignment
    - Outlier cleaning using IQR
    - Savitzky-Golay filtering
    - Rescaling

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **centered_reference_keypoint** (`str, optional`): Keypoint to use as centered reference.
* **orientation_reference_keypoint** (`str, optional`): Keypoint to use as orientation reference.
* **run_lowconf_cleaning** (`bool, optional`): Whether to run low confidence cleaning.
* **run_egocentric_alignment** (`bool, optional`): Whether to run egocentric alignment.
* **run_outlier_cleaning** (`bool, optional`): Whether to run outlier cleaning.
* **run_savgol_filtering** (`bool, optional`): Whether to run Savitzky-Golay filtering.
* **run_rescaling** (`bool, optional`): Whether to run rescaling. Defaults to True.
* **save_logs** (`bool, optional`): Whether to save logs.

**Returns**

* `variable name of the last-executed preprocessing step output, also saved`
* `to the config as ``preprocessed_variable```

#### preprocess\_session

```python
def preprocess_session(file_path: str, steps: List[Tuple[Callable,
                                                         dict]]) -> None
```

Run preprocessing steps on one session&#x27;s processed file: read once, apply the
steps in order, and replace the file once.

**Parameters**

* **file_path** (`str`): Path to the session&#x27;s processed netCDF file.
* **steps** (`list of (function, kwargs)`): Dataset functions to apply in order, each called as ``function(ds=ds, **kwargs)``.

**Returns**

* `None`

