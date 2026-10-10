---
sidebar_label: scaling
title: preprocessing.scaling
---

#### logger\_config

#### logger

#### rescaling

```python
def rescaling(config: dict,
              read_from_variable: str = "position_processed",
              save_to_variable: str = "position_scaled",
              save_logs: bool = True) -> None
```

Rescale the position data by dividing by the individual scale values.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **read_from_variable** (`str, optional`): Variable to read from the dataset.
* **save_to_variable** (`str, optional`): Variable to save the rescaled data to.
* **save_logs** (`bool, optional`): Whether to save logs.

**Returns**

* `None`

#### rescale\_dataset

```python
def rescale_dataset(ds: xr.Dataset, read_from_variable: str,
                    save_to_variable: str) -> bool
```

Divide one session&#x27;s positions by each individual&#x27;s ``individual_scale``, in place.

**Parameters**

* **ds** (`xr.Dataset`): Session dataset.
* **read_from_variable** (`str`): Variable to read from the dataset.
* **save_to_variable** (`str`): Variable to save the rescaled data to.

**Returns**

* `bool`: False if the dataset has no ``individual_scale`` and was left unchanged.

