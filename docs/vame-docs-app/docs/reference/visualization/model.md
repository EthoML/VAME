---
sidebar_label: model
title: visualization.model
---

#### use\_gpu

#### plot\_reconstruction

```python
def plot_reconstruction(filepath: str,
                        test_loader: Data.DataLoader,
                        seq_len_half: int,
                        model: RNN_VAE,
                        model_name: str,
                        FUTURE_DECODER: bool,
                        FUTURE_STEPS: int,
                        suffix: Optional[str] = None,
                        show_figure: bool = False) -> None
```

Plot the reconstruction and future prediction of the input sequence.
Saves the plot to:
- project_name/
    - model/
        - evaluate/
            - Reconstruction_model_name.png

**Parameters**

* **filepath** (`str`): Path to save the plot.
* **test_loader** (`Data.DataLoader`): DataLoader for the test dataset.
* **seq_len_half** (`int`): Half of the temporal window size.
* **model** (`RNN_VAE`): Trained VAE model.
* **model_name** (`str`): Name of the model.
* **FUTURE_DECODER** (`bool`): Flag indicating whether the model has a future prediction decoder.
* **FUTURE_STEPS** (`int`): Number of future steps to predict.
* **suffix** (`str, optional`): Suffix for the saved plot filename. Defaults to None.
* **show_figure** (`bool, optional`): Flag indicating whether to show the plot. Defaults to False.

**Returns**

* `None`

#### plot\_loss

```python
def plot_loss(config: dict,
              model_name: str,
              save_to_file: bool = False,
              show_figure: bool = True) -> None
```

Plot the losses of the trained model.
Saves the plot to:
- project_name/
    - model/
        - evaluate/
            - MSE-and-KL-Loss_model_name.png

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **model_name** (`str`): Name of the model.
* **save_to_file** (`bool, optional`): Flag indicating whether to save the plot. Defaults to False.
* **show_figure** (`bool, optional`): Flag indicating whether to show the plot. Defaults to True.

**Returns**

* `None`

