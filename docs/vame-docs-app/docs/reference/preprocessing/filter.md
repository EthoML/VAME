---
sidebar_label: filter
title: preprocessing.filter
---

#### logger\_config

#### logger

#### savgol\_filtering

```python
def savgol_filtering(config: dict,
                     read_from_variable: str = "position_processed",
                     save_to_variable: str = "position_processed",
                     save_logs: bool = True) -> None
```

Apply Savitzky-Golay filter to the data.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **read_from_variable** (`str, optional`): Variable to read from the dataset.
* **save_to_variable** (`str, optional`): Variable to save the filtered data to.
* **save_logs** (`bool, optional`): Whether to save logs.

**Returns**

* `None`

