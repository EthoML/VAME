---
sidebar_label: csv_to_npy
title: util.csv_to_npy
---

#### logger\_config

#### logger

#### pose\_to\_numpy

```python
def pose_to_numpy(config: dict, save_logs=False) -> None
```

Converts a pose-estimation.csv file to a numpy array.
Note that this code is only useful for data which is a priori egocentric, i.e. head-fixed
or otherwise restrained animals.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **save_logs** (`bool, optional`): If True, the logs will be saved to a file, by default False.

**Raises**

* `ValueError`: If the config.yaml file indicates that the data is not egocentric.

