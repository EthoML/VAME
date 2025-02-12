---
sidebar_label: video
title: video.video
---

#### logger\_config

#### logger

#### get\_video\_frame\_rate

```python
def get_video_frame_rate(video_path)
```

#### add\_videos\_to\_project

```python
def add_videos_to_project(config: dict,
                          videos: List[str],
                          copy_videos: bool = False) -> None
```

Add videos to the project, ensuring video stems match session names.

**Parameters**

* **config** (`dict`): Configuration settings for the project.
* **videos** (`List[str]`): List of video file paths to add.
* **copy_videos** (`bool, optional`): If True, copy videos to project directory. If False, create symbolic links, by default False

**Raises**

* `ValueError`: If video stems don&#x27;t match session names.

