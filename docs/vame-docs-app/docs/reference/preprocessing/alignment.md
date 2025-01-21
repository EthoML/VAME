---
sidebar_label: alignment
title: preprocessing.alignment
---

#### logger\_config

#### logger

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

