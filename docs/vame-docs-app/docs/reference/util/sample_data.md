---
sidebar_label: sample_data
title: util.sample_data
---

#### logger\_config

#### logger

#### T

#### SAMPLE\_DATA\_URL

#### DOWNLOAD\_PATH

#### DATASETS

#### GIN\_UNREACHABLE\_MSG

#### \_with\_retries

```python
def _with_retries(func: Callable[[], T],
                  attempts: int = 4,
                  base_delay: float = 5.0) -> T
```

Call func, retrying on network errors with exponential backoff.

#### \_fetch

```python
def _fetch(remote_path: str,
           sha256: str | None,
           fname: str | None = None) -> Path
```

Download one file from the sample-data repository, or reuse the cached copy if its hash matches.

#### download\_sample\_data

```python
def download_sample_data(source_software: str,
                         with_video: bool = True) -> dict
```

Download sample data.

Files are downloaded from movement&#x27;s sample-data repository on SWC GIN and cached in
~/.movement/data. This bypasses `movement.sample_data` for now: up to movement 0.17.0 it
downloads from G-Node GIN (gin.g-node.org), which is often unreachable. movement moved to
SWC GIN in https://github.com/neuroinformatics-unit/movement/pull/1080, which is not yet
released. Once a movement release includes it, go back to
`movement.sample_data.fetch_dataset_paths` and require that version.

**Parameters**

* **source_software** (`str`): Source software used for pose estimation.
* **with_video** (`bool, optional`): If True, the video will be downloaded as well. Defaults to True.

**Returns**

* `dict`: Dictionary with the paths to the downloaded sample data (&quot;poses&quot;, &quot;video&quot;, &quot;frame&quot;)
and the video frame rate (&quot;fps&quot;). The video is saved under the pose file&#x27;s name.

