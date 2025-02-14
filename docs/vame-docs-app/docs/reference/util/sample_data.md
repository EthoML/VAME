---
sidebar_label: sample_data
title: util.sample_data
---

#### DOWNLOAD\_PATH

#### dataset\_options

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

