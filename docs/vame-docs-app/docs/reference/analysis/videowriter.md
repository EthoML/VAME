---
sidebar_label: videowriter
title: analysis.videowriter
---

#### logger\_config

#### logger

#### create\_cluster\_videos

```python
def create_cluster_videos(
        config: dict,
        path_to_file: str,
        session: str,
        n_clusters: int,
        video_type: str,
        flag: str,
        segmentation_algorithm: SegmentationAlgorithms,
        output_video_type: str = ".mp4",
        tqdm_logger_stream: Union[TqdmToLogger, None] = None) -> None
```

Generate cluster videos and save them to filesystem on project folder.

**Parameters**

* **config** (`dict`): Configuration parameters.
* **path_to_file** (`str`): Path to the file.
* **session** (`str`): Name of the session.
* **n_clusters** (`int`): Number of clusters.
* **video_type** (`str`): Type of input video.
* **flag** (`str`): Flag indicating the type of video (motif or community).
* **segmentation_algorithm** (`SegmentationAlgorithms`): Which segmentation algorithm to use. Options are &#x27;hmm&#x27; or &#x27;kmeans&#x27;.
* **output_video_type** (`str, optional`): Type of output video. Default is &#x27;.mp4&#x27;.
* **tqdm_logger_stream** (`TqdmToLogger, optional`): Tqdm logger stream. Default is None.

**Returns**

* `None`

#### motif\_videos

```python
@save_state(model=MotifVideosFunctionSchema)
def motif_videos(config: dict,
                 video_type: str = ".mp4",
                 output_video_type: str = ".mp4",
                 save_logs: bool = True) -> None
```

Generate motif videos and save them to filesystem.
Fills in the values in the &quot;motif_videos&quot; key of the states.json file.
Files are saved at:
- project_name/
    - results/
        - session_name/
            - model_name/
                - segmentation_algorithm-n_clusters/
                    - cluster_videos/
                        - session_name-motif_0.mp4
                        - session_name-motif_1.mp4
                        - ...

**Parameters**

* **config** (`dict`): Configuration parameters.
* **segmentation_algorithm** (`SegmentationAlgorithms`): Which segmentation algorithm to use. Options are &#x27;hmm&#x27; or &#x27;kmeans&#x27;.
If None, it will be taken from the config file.
* **video_type** (`str, optional`): Type of video. Default is &#x27;.mp4&#x27;.
* **output_video_type** (`str, optional`): Type of output video. Default is &#x27;.mp4&#x27;.
* **save_logs** (`bool, optional`): Save logs to filesystem. Default is True.

**Returns**

* `None`

#### community\_videos

```python
@save_state(model=CommunityVideosFunctionSchema)
def community_videos(config: dict,
                     video_type: str = ".mp4",
                     output_video_type: str = ".mp4",
                     save_logs: bool = True) -> None
```

Generate community videos and save them to filesystem on project community_videos folder.
Fills in the values in the &quot;community_videos&quot; key of the states.json file.
Files are saved at:
- project_name/
    - results/
        - session_name/
            - model_name/
                - segmentation_algorithm-n_clusters/
                    - community_videos/
                        - session_name-community_0.mp4
                        - session_name-community_1.mp4
                        - ...

**Parameters**

* **config** (`dict`): Configuration parameters.
* **video_type** (`str, optional`): Type of video. Default is &#x27;.mp4&#x27;.
* **output_video_type** (`str, optional`): Type of output video. Default is &#x27;.mp4&#x27;.
* **save_logs** (`bool, optional`): Save logs to filesystem. Default is True.

**Returns**

* `None`

