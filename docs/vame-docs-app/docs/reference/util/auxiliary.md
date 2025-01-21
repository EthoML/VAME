---
sidebar_label: auxiliary
title: util.auxiliary
---

#### create\_config\_template

```python
def create_config_template() -> Tuple[dict, ruamel.yaml.YAML]
```

Creates a template for the config.yaml file.

**Returns**

* `Tuple[dict, ruamel.yaml.YAML]`: A tuple containing the template dictionary and the Ruamel YAML instance.

#### read\_config

```python
def read_config(config_file: str) -> dict
```

Reads structured config file defining a project.

**Parameters**

* **config_file** (`str`): Path to the config file.

**Returns**

* `dict`: The contents of the config file as a dictionary.

#### write\_config

```python
def write_config(configname: str, cfg: dict) -> None
```

Write structured config file.

**Parameters**

* **configname** (`str`): Path to the config file.
* **cfg** (`dict`): Dictionary containing the config data.

#### read\_states

```python
def read_states(config: dict) -> dict
```

Reads the states.json file.

**Parameters**

* **config** (`dict`): Dictionary containing the config data.

**Returns**

* `dict`: The contents of the states.json file as a dictionary.

