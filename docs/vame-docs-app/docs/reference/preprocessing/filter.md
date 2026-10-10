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

#### savgol\_filter\_dataset

```python
def savgol_filter_dataset(ds: xr.Dataset, savgol_length: int,
                          savgol_order: int, read_from_variable: str,
                          save_to_variable: str) -> None
```

Apply a Savitzky-Golay filter to one session&#x27;s dataset, in place.

**Parameters**

* **ds** (`xr.Dataset`): Session dataset.
* **savgol_length** (`int`): Filter window length.
* **savgol_order** (`int`): Polynomial order.
* **read_from_variable** (`str`): Variable to read from the dataset.
* **save_to_variable** (`str`): Variable to save the filtered data to.

**Returns**

* `None`

