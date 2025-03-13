---
sidebar_label: auxiliary
title: util.auxiliary
---

#### get\_version

```python
def get_version() -> str
```

Gets the VAME package version from pyproject.toml.

**Returns**

* `str`: The version string.

#### \_convert\_enums\_to\_values

```python
def _convert_enums_to_values(obj: Any) -> Any
```

Recursively converts enum values to their string representations.

**Parameters**

* **obj** (`Any`): The object to convert.

**Returns**

* `Any`: The converted object with enum values replaced by their string representations.

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
def write_config(config_path: str, config: dict) -> None
```

Write structured config file.

**Parameters**

* **config_path** (`str`): Path to the config file.
* **config** (`dict`): Dictionary containing the config data.

#### read\_states

```python
def read_states(config: dict) -> dict
```

Reads the states.json file.

**Parameters**

* **config** (`dict`): Dictionary containing the config data.

**Returns**

* `dict`: The contents of the states.json file as a dictionary.

