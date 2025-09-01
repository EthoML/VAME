---
sidebar_label: gif_pose_helper
title: util.gif_pose_helper
---

#### logger\_config

#### logger

#### get\_animal\_frames

```python
def get_animal_frames(
    config: dict,
    session: str,
    pose_ref_index: list,
    start: int,
    length: int,
    subtract_background: bool,
    file_format: str = ".mp4",
    crop_size: tuple = (300, 300)) -> list
```

Extracts frames of an animal from a video file and returns them as a list.

**Parameters**

* **config** (`dict`): Configuration dictionary containing project information.
* **session** (`str`): Name of the session.
* **pose_ref_index** (`list`): List of reference coordinate indices for alignment.
* **start** (`int`): Starting frame index.
* **length** (`int`): Number of frames to extract.
* **subtract_background** (`bool`): Whether to subtract background or not.
* **file_format** (`str, optional`): Format of the video file. Defaults to &#x27;.mp4&#x27;.
* **crop_size** (`tuple, optional`): Size of the cropped area. Defaults to (300, 300).

**Returns**

* `list:`: List of extracted frames.

