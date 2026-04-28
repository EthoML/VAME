---
sidebar_label: extra
title: io.extra
---

#### logger\_config

#### logger

#### add\_extra\_features

```python
def add_extra_features(config: dict,
                       session: str,
                       features: Dict[str, np.ndarray],
                       register_in_config: bool = True,
                       config_path: Optional[str] = None) -> None
```

Append pre-computed extra (non-pose) features to a session&#x27;s processed .nc.

Each value in ``features`` must be a 1-D numpy array whose length matches
the session&#x27;s ``position_processed`` time dimension. Features are written
as data variables with dims ``(&quot;time&quot;, &quot;individuals&quot;)`` and populated only
at ``individuals[0]`` (multi-individual support deferred); other
individuals are filled with NaN.

The write is atomic: data is written to a temporary ``.nc`` file alongside
the original, then renamed into place, so a failure mid-write cannot
corrupt the existing file.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **session** (`str`): Session name (must match an entry in ``config[&quot;session_names&quot;]``).
* **features** (`dict[str, np.ndarray]`): Mapping of feature name to a 1-D numeric array of length
``len(position_processed.time)``.
* **register_in_config** (`bool, default True`): If True, append the feature names to ``config[&quot;extra_features&quot;]``
(de-duplicated, order-preserved) and persist the config to disk.
* **config_path** (`str, optional`): Explicit path to ``config.yaml``. If omitted, defaults to
``{config[&quot;project_path&quot;]}/config.yaml``.

**Returns**

* `None`

**Raises**

* `FileNotFoundError`: If the session&#x27;s processed ``.nc`` does not exist.
* `ValueError`: If a feature array is not 1-D, has the wrong length, or is non-numeric.

