---
sidebar_label: model
title: visualization.model
---

#### plot\_loss

```python
def plot_loss(config: dict,
              model_name: Optional[str] = None,
              save_to_file: bool = False,
              show_figure: bool = True) -> None
```

Plot the losses of the trained model.
Saves the plot to:
- project_name/
    - model/
        - evaluate/
            - mse_and_kl_loss_model_name.png

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **model_name** (`str`): Name of the model. Defaults to None, in which case the model name in config is used.
* **save_to_file** (`bool, optional`): Flag indicating whether to save the plot. Defaults to False.
* **show_figure** (`bool, optional`): Flag indicating whether to show the plot. Defaults to True.

**Returns**

* `None`

