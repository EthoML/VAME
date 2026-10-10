---
sidebar_label: cleaning
title: preprocessing.cleaning
---

#### logger\_config

#### logger

#### lowconf\_cleaning

```python
def lowconf_cleaning(config: dict,
                     read_from_variable: str = "position_processed",
                     save_to_variable: str = "position_processed",
                     save_logs: bool = True) -> None
```

Clean the low confidence data points from the dataset. Processes position data by:
 - setting low-confidence points to NaN
 - interpolating NaN points

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **read_from_variable** (`str, optional`): Variable to read from the dataset.
* **save_to_variable** (`str, optional`): Variable to save the cleaned data to.
* **save_logs** (`bool, optional`): Whether to save logs.

**Returns**

* `None`

#### lowconf\_clean\_dataset

```python
def lowconf_clean_dataset(ds: xr.Dataset, pose_confidence: float,
                          read_from_variable: str,
                          save_to_variable: str) -> None
```

Set low-confidence points of one session&#x27;s dataset to NaN and interpolate them, in place.

**Parameters**

* **ds** (`xr.Dataset`): Session dataset.
* **pose_confidence** (`float`): Confidence threshold.
* **read_from_variable** (`str`): Variable to read from the dataset.
* **save_to_variable** (`str`): Variable to save the cleaned data to.

**Returns**

* `None`

#### outlier\_cleaning

```python
def outlier_cleaning(config: dict,
                     read_from_variable: str = "position_processed",
                     save_to_variable: str = "position_processed",
                     save_logs: bool = True) -> None
```

Clean the outliers from the dataset. Processes position data by:
 - setting outlier points to NaN
 - interpolating NaN points

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **read_from_variable** (`str, optional`): Variable to read from the dataset.
* **save_to_variable** (`str, optional`): Variable to save the cleaned data to.
* **save_logs** (`bool, optional`): Whether to save logs.

**Returns**

* `None`

#### outlier\_clean\_dataset

```python
def outlier_clean_dataset(ds: xr.Dataset, robust: bool, iqr_factor: float,
                          read_from_variable: str,
                          save_to_variable: str) -> None
```

Set IQR outliers of one session&#x27;s dataset to NaN and interpolate them, in place.

**Parameters**

* **ds** (`xr.Dataset`): Session dataset.
* **robust** (`bool`): Whether to clean outliers. If False, the data is copied unchanged.
* **iqr_factor** (`float`): IQR multiplier for the outlier cutoff.
* **read_from_variable** (`str`): Variable to read from the dataset.
* **save_to_variable** (`str`): Variable to save the cleaned data to.

**Returns**

* `None`

