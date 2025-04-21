---
sidebar_label: alignment
title: preprocessing.alignment
---

#### logger\_config

#### logger

#### calculate\_geometric\_distance

```python
def calculate_geometric_distance(positions, keypoint1_idx, keypoint2_idx)
```

Calculate the geometric distance between two keypoints.
Parameters
----------
positions : np.ndarray
    Array of positions with shape (time, space, keypoints, individuals) or (time, space, keypoints)
keypoint1_idx : int
    Index of the first keypoint
keypoint2_idx : int
    Index of the second keypoint
Returns
-------
np.ndarray
    Array of distances with shape (time, individuals) or (time,)

#### egocentrically\_align\_and\_center

```python
def egocentrically_align_and_center(
        config: dict,
        centered_reference_keypoint: str = "snout",
        orientation_reference_keypoint: str = "tailbase",
        read_from_variable: str = "position_processed",
        save_to_variable: str = "position_egocentric_aligned") -> None
```

Aligns the time series by first centralizing all positions around the first keypoint
and then applying rotation to align with the line connecting the two keypoints.

**Parameters**

* **config** (`dict`): Configuration dictionary
* **centered_reference_keypoint** (`str`): Name of the keypoint to use as centered reference.
* **orientation_reference_keypoint** (`str`): Name of the keypoint to use as orientation reference.

**Returns**

* `None`

