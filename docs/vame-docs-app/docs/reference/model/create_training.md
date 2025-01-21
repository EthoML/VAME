---
sidebar_label: create_training
title: model.create_training
---

#### logger\_config

#### logger

#### traindata\_aligned

```python
def traindata_aligned(config: dict,
                      sessions: List[str] | None = None,
                      test_fraction: float | None = None,
                      read_from_variable: str = "position_processed") -> None
```

Create training dataset for aligned data.
Save numpy arrays with the test/train info to the project folder.

**Parameters**

* **config** (`dict`): Configuration parameters dictionary.
* **sessions** (`List[str], optional`): List of session names. If None, all sessions will be used. Defaults to None.
* **test_fraction** (`float, optional`): Fraction of data to use as test data. Defaults to 0.1.

**Returns**

* `None`

#### create\_trainset

```python
@save_state(model=CreateTrainsetFunctionSchema)
def create_trainset(config: dict, save_logs: bool = False) -> None
```

Creates a training and test datasets for the VAME model.
Fills in the values in the &quot;create_trainset&quot; key of the states.json file.
Creates the training dataset for VAME at:
- project_name/
    - data/
        - session00/
            - session00-PE-seq-clean.npy
        - session01/
            - session01-PE-seq-clean.npy
        - train/
            - test_seq.npy
            - train_seq.npy

The produced -clean.npy files contain the aligned time series data in the
shape of (num_dlc_features - 2, num_video_frames).

The produced test_seq.npy contains the combined data in the shape of (num_dlc_features - 2, num_video_frames * test_fraction).

The produced train_seq.npy contains the combined data in the shape of (num_dlc_features - 2, num_video_frames * (1 - test_fraction)).

**Parameters**

* **config** (`dict`): Configuration parameters dictionary.
* **save_logs** (`bool, optional`): If True, the function will save logs to the project folder. Defaults to False.

**Returns**

* `None`

