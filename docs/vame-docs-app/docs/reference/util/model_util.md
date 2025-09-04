---
sidebar_label: model_util
title: util.model_util
---

#### logger\_config

#### logger

#### load\_training\_metadata

```python
def load_training_metadata(config: dict) -> dict
```

Load training metadata to get keypoints used during training.

**Parameters**

* **config** (`dict`): Configuration dictionary.

**Returns**

* `dict`: Training metadata containing keypoints_used and other parameters.

**Raises**

* `FileNotFoundError`: If metadata.json file is not found.
* `ValueError`: If metadata is invalid or missing required fields.

#### load\_model

```python
def load_model(config: dict, model_name: str, fixed: bool = True) -> RNN_VAE
```

Load the VAME model.

Args:
    config (dict): Configuration dictionary.
    model_name (str): Name of the model.
    fixed (bool): Fixed or variable length sequences.

Returns
    RNN_VAE: Loaded VAME model.

