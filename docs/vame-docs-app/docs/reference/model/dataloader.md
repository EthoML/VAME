---
sidebar_label: dataloader
title: model.dataloader
---

## SEQUENCE\_DATASET Objects

```python
class SEQUENCE_DATASET(Dataset)
```

#### \_\_init\_\_

```python
def __init__(path_to_file: str, data: str, train: bool, temporal_window: int,
             **kwargs) -> None
```

Initialize the Sequence Dataset.
Creates files at:
- project_name/
- data/
    - train/
        - seq_mean.npy
        - seq_std.npy

**Parameters**

* **path_to_file** (`str`): Path to the dataset files.
* **data** (`str`): Name of the data file.
* **train** (`bool`): Flag indicating whether it&#x27;s training data.
* **temporal_window** (`int`): Size of the temporal window.

**Returns**

* `None`

#### \_\_len\_\_

```python
def __len__() -> int
```

Return the number of data points.

**Returns**

* `int`: Number of data points.

#### \_\_getitem\_\_

```python
def __getitem__(index: int) -> torch.Tensor
```

Get a normalized sequence at the specified index.

**Parameters**

* **index** (`int`): Index of the item.

**Returns**

* `torch.Tensor`: Normalized sequence data at the specified index.

