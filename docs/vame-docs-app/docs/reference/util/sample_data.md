---
sidebar_label: sample_data
title: util.sample_data
---

#### logger\_config

#### logger

#### T

#### GIN\_UNREACHABLE\_MSG

#### \_with\_retries

```python
def _with_retries(func: Callable[[], T],
                  attempts: int = 4,
                  base_delay: float = 5.0) -> T
```

Call func, retrying on network errors with exponential backoff.

#### \_import\_movement\_sample\_data

```python
def _import_movement_sample_data()
```

#### download\_sample\_data

```python
def download_sample_data(source_software: str,
                         with_video: bool = True) -> dict
```

Download sample data.

**Parameters**

* **source_software** (`str`): Source software used for pose estimation.
* **with_video** (`bool, optional`): If True, the video will be downloaded as well. Defaults to True.

**Returns**

* `dict`: Dictionary with the paths to the downloaded sample data.

