---
sidebar_label: extra
title: preprocessing.extra
---

#### logger\_config

#### logger

#### validate\_extra\_features

```python
def validate_extra_features(
        config: dict,
        sessions: Optional[List[str]] = None,
        extra_features: Optional[List[str]] = None) -> None
```

Validate that every session&#x27;s processed .nc satisfies the extra-feature contract.

For each name in the extra-feature list and each session, checks:
  - The processed ``.nc`` file exists.
  - The data variable is present.
  - Its dims are ``(time,)`` or ``(time, individuals)``.
  - Its ``time`` length matches the session&#x27;s ``position_processed`` time length.
  - Its dtype is numeric (float or int).
  - The first-individual slice contains no NaN or Inf values.

All problems across all sessions are collected and reported in a single
``ValueError`` so users can fix them in one pass.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **sessions** (`list of str, optional`): Sessions to validate. Defaults to ``config[&quot;session_names&quot;]``.
* **extra_features** (`list of str, optional`): Override the list of feature names to validate. Defaults to
``config[&quot;extra_features&quot;]``. Inference paths pass the list recovered
from training metadata so they validate against what the model expects,
not what the live config currently contains.

**Returns**

* `None`: Returns silently on success.

**Raises**

* `ValueError`: If any registered extra feature fails the contract for any session.

