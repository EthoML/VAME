---
sidebar_label: video
title: video.video
---

#### logger\_config

#### logger

#### VIDEO\_SUFFIXES

#### is\_video\_file

```python
def is_video_file(path: Union[str, Path]) -> bool
```

Whether ``path`` looks like a video VAME can read, by extension.

#### resolve\_video\_type

```python
def resolve_video_type(videos: List[Union[str, Path]]) -> Optional[str]
```

The extension shared by ``videos``, or None when they are not all the same.

Recorded in the config as the project&#x27;s video format. A mixed-format project
has no single answer, so callers fall back to resolving each session&#x27;s video
from disk.

#### get\_session\_video\_path

```python
def get_session_video_path(config: dict, session: str) -> Path
```

Resolve the raw video belonging to ``session``.

Videos are linked into ``data/raw`` under their session name, keeping their
original extension, so the format is read back off disk rather than assumed.
``video_type`` in the config only breaks ties when a session has more than
one video file.

**Parameters**

* **config** (`dict`): Configuration parameters.
* **session** (`str`): Name of the session.

**Returns**

* `Path`: Path to the session&#x27;s video file.

**Raises**

* `FileNotFoundError`: If the session has no video file in ``data/raw``.

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

Records the resulting format as ``video_type`` in the project&#x27;s config.

**Parameters**

* **config** (`dict`): Configuration settings for the project.
* **videos** (`List[str]`): List of video file paths to add.
* **copy_videos** (`bool, optional`): If True, copy videos to project directory. If False, create symbolic links, by default False

**Raises**

* `ValueError`: If video stems don&#x27;t match session names, or a video is not a supported format.

