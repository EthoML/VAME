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

