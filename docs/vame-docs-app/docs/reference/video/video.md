---
sidebar_label: video
title: video.video
---

#### get\_video\_frame\_rate

```python
def get_video_frame_rate(video_path)
```

#### extract\_session\_number

```python
def extract_session_number(filename: str,
                           pattern=r"(?:Session)0*(\d+)\D") -> str | None
```

Takes a filename and finds a match based on a regex pattern string

**Parameters**

* **filename** (`str`): String to search
* **pattern** (`regexp, optional`): regex expression for the search pattern, by default r&#x27;(?:Session)0*(\d+)\D&#x27;

**Returns**

* `literal eval return`: returns the capture group from the regex expression as the literal eval type.

#### find\_matching\_session\_files

```python
def find_matching_session_files(files: list, session_num) -> list
```

Using a regex pattern, finds all files that match the session number.

**Parameters**

* **files** (`list`): list of file paths to search through
* **session_num**: number to search for

**Returns**

* `list`: list of files with the matching session number

