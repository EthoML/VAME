---
sidebar_label: preprocessing
title: preprocessing.preprocessing
---

#### logger\_config

#### logger

#### preprocessing

```python
def preprocessing(config: dict,
                  centered_reference_keypoint: str = "snout",
                  orientation_reference_keypoint: str = "tailbase",
                  save_logs: bool = False) -> None
```

Preprocess the data by:
    - Cleaning low confidence data points
    - Egocentric alignment
    - Outlier cleaning
    - Savitzky-Golay filtering

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **centered_reference_keypoint** (`str, optional`): Keypoint to use as centered reference.
* **orientation_reference_keypoint** (`str, optional`): Keypoint to use as orientation reference.

**Returns**

* `None`

