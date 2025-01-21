---
sidebar_label: model_util
title: util.model_util
---

#### logger\_config

#### logger

#### load\_model

```python
def load_model(cfg: dict, model_name: str, fixed: bool = True) -> RNN_VAE
```

Load the VAME model.

Args:
    cfg (dict): Configuration dictionary.
    model_name (str): Name of the model.
    fixed (bool): Fixed or variable length sequences.

Returns
    RNN_VAE: Loaded VAME model.

