---
sidebar_label: gif_creator
title: analysis.gif_creator
---

#### logger\_config

#### logger

#### create\_video

```python
def create_video(path_to_file: str, session: str, embed: np.ndarray,
                 clabel: np.ndarray, frames: List[np.ndarray], start: int,
                 length: int, max_lag: int, num_points: int) -> None
```

Create video frames for the given embedding.

**Parameters**

* **path_to_file** (`str`): Path to the file.
* **session** (`str`): Session name.
* **embed** (`np.ndarray`): Embedding array.
* **clabel** (`np.ndarray`): Cluster labels.
* **frames** (`List[np.ndarray]`): List of frames.
* **start** (`int`): Starting index.
* **length** (`int`): Length of the video.
* **max_lag** (`int`): Maximum lag.
* **num_points** (`int`): Number of points.

**Returns**

* `None`

#### gif

```python
def gif(
    config: str,
    pose_ref_index: list,
    segmentation_algorithm: SegmentationAlgorithms,
    subtract_background: bool = True,
    start: int | None = None,
    length: int = 500,
    max_lag: int = 30,
    label: str = "community",
    file_format: str = ".mp4",
    crop_size: Tuple[int, int] = (300, 300)) -> None
```

Create a GIF from the given configuration.

**Parameters**

* **config** (`str`): Path to the configuration file.
* **pose_ref_index** (`list`): List of reference coordinate indices for alignment.
* **segmentation_algorithm** (`SegmentationAlgorithms`): Segmentation algorithm.
* **subtract_background** (`bool, optional`): Whether to subtract background. Defaults to True.
* **start :int, optional**: Starting index. Defaults to None.
* **length** (`int, optional`): Length of the video. Defaults to 500.
* **max_lag** (`int, optional`): Maximum lag. Defaults to 30.
* **label** (`str, optional`): Label type [None, community, motif]. Defaults to &#x27;community&#x27;.
* **file_format** (`str, optional`): File format. Defaults to &#x27;.mp4&#x27;.
* **crop_size** (`Tuple[int, int], optional`): Crop size. Defaults to (300,300).

**Returns**

* `None`

