---
sidebar_label: evaluate
title: model.evaluate
---

#### logger\_config

#### logger

#### use\_gpu

#### eval\_temporal

```python
def eval_temporal(config: dict,
                  use_gpu: bool,
                  model_name: str,
                  fixed: bool,
                  snapshot: Optional[str] = None,
                  suffix: Optional[str] = None) -> None
```

Evaluate the temporal aspects of the trained model.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **use_gpu** (`bool`): Flag indicating whether to use GPU for evaluation.
* **model_name** (`str`): Name of the model.
* **fixed** (`bool`): Flag indicating whether the data is fixed or not.
* **snapshot** (`str, optional`): Path to the model snapshot. Defaults to None.
* **suffix** (`str, optional`): Suffix for the saved plot filename. Defaults to None.

**Returns**

* `None`

#### evaluate\_model

```python
@save_state(model=EvaluateModelFunctionSchema)
def evaluate_model(config: dict,
                   use_snapshots: bool = False,
                   save_logs: bool = True) -> None
```

Evaluate the trained model.
Fills in the values in the &quot;evaluate_model&quot; key of the states.json file.
Saves the evaluation results to:
- project_name/
    - model/
        - evaluate/

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **use_snapshots** (`bool, optional`): Whether to plot for all snapshots or only the best model. Defaults to False.
* **save_logs** (`bool, optional`): Whether to save logs. Defaults to True.

**Returns**

* `None`

