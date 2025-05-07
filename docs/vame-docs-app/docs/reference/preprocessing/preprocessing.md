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
                  run_rescaling: bool = False,
                  save_logs: bool = True) -> None
```

Preprocess the data by:
    - Cleaning low confidence data points
    - Egocentric alignment
    - Outlier cleaning using IQR
    - Rescaling
    - Savitzky-Golay filtering

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **centered_reference_keypoint** (`str, optional`): Keypoint to use as centered reference.
* **orientation_reference_keypoint** (`str, optional`): Keypoint to use as orientation reference.
* **run_lowconf_cleaning** (`bool, optional`): Whether to run low confidence cleaning.
* **run_egocentric_alignment** (`bool, optional`): Whether to run egocentric alignment.
* **run_outlier_cleaning** (`bool, optional`): Whether to run outlier cleaning.
* **run_savgol_filtering** (`bool, optional`): Whether to run Savitzky-Golay filtering.
* **run_rescaling** (`bool, optional`): Whether to run rescaling.
* **save_logs** (`bool, optional`): Whether to save logs.

**Returns**

* `None`

