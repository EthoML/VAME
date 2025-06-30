---
sidebar_label: inspection
title: preprocessing.inspection
---

#### pose\_estimation\_inspection

```python
def pose_estimation_inspection(config,
                               read_from_variable: str = "position_raw",
                               save_to_file: bool = False,
                               show_figure: bool = True) -> None
```

Inspect pose estimation data for quality and completeness.

**Parameters**

* **config** (`dict`): Configuration parameters.
* **read_from_variable** (`str`): Name of the variable to read the raw position data from.

**Returns**

* `None`

#### plot\_pose\_estimation\_inspection\_matplotlib

```python
def plot_pose_estimation_inspection_matplotlib(
        config,
        confidence_data: np.ndarray,
        keypoint_names: np.ndarray,
        save_to_file: bool = False,
        show_figure: bool = True) -> None
```

Plot pose estimation inspection results using matplotlib with multiple subplots.

**Parameters**

* **config** (`dict`): Configuration parameters.
* **confidence_data** (`np.ndarray`): Confidence data array with shape (time, keypoints, individuals).
* **keypoint_names** (`np.ndarray`): Array of keypoint names.

**Returns**

* `None`

