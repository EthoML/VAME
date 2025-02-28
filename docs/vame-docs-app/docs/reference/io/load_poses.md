---
sidebar_label: load_poses
title: io.load_poses
---

#### load\_pose\_estimation

```python
def load_pose_estimation(pose_estimation_file: Path | str,
                         source_software: Literal["DeepLabCut", "SLEAP",
                                                  "LightningPose"],
                         video_file: Optional[Path | str] = None,
                         fps: Optional[float] = None) -> xr.Dataset
```

Load pose estimation data.

**Parameters**

* **pose_estimation_file** (`Path or str`): Path to the pose estimation file.
* **video_file** (`Path or str`): Path to the video file.
* **fps** (`float, optional`): Sampling rate of the video.
* **source_software** (`Literal["DeepLabCut", "SLEAP", "LightningPose"]`): Source software used for pose estimation.

**Returns**

* **ds** (`xarray.Dataset`): Pose estimation dataset.

#### load\_vame\_dataset

```python
def load_vame_dataset(ds_path: Path | str) -> xr.Dataset
```

Load VAME dataset.

**Parameters**

* **ds_path** (`Path or str`): Path to the netCDF dataset.

**Returns**

* `xr.Dataset`: VAME dataset

#### nc\_to\_dataframe

```python
def nc_to_dataframe(nc_data)
```

#### read\_pose\_estimation\_file

```python
def read_pose_estimation_file(
    file_path: str,
    file_type: Optional[Literal["csv", "nwb", "slp", "h5"]] = None,
    path_to_pose_nwb_series_data: Optional[str] = None
) -> Tuple[pd.DataFrame, np.ndarray, xr.Dataset]
```

Read pose estimation file.

**Parameters**

* **file_path** (`str`): Path to the pose estimation file.
* **file_type** (`PoseEstimationFiletype`): Type of the pose estimation file. Supported types are &#x27;csv&#x27; and &#x27;nwb&#x27;.
* **path_to_pose_nwb_series_data** (`str, optional`): Path to the pose data inside the nwb file, by default None

**Returns**

* `Tuple[pd.DataFrame, np.ndarray]`: Tuple containing the pose estimation data as a pandas DataFrame and a numpy array.

