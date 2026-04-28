---
sidebar_label: load_poses
title: io.load_poses
---

#### \_validate\_movement\_schema

```python
def _validate_movement_schema(ds: xr.Dataset) -> None
```

Validate that a Dataset matches the movement-format pose schema VAME expects.

Aggregates all problems into a single ``ValueError`` so users see the full
set of issues at once instead of fixing them one at a time.

#### load\_pose\_estimation

```python
def load_pose_estimation(
        pose_estimation_file: Path | str,
        source_software: Literal["DeepLabCut", "SLEAP", "LightningPose", "NWB",
                                 "auto", "movement"] = "auto",
        video_file: Optional[Path | str] = None,
        fps: Optional[float] = None,
        processing_module_key: str = "behavior",
        pose_estimation_key: str = "PoseEstimation") -> xr.Dataset
```

Load pose estimation data.

**Parameters**

* **pose_estimation_file** (`Path or str`): Path to the pose estimation file. Dispatched through movement&#x27;s
unified loader, which auto-detects format from extension and contents.
* **source_software** (`str, optional`): Source software used for pose estimation. Defaults to ``&quot;auto&quot;``, which
lets movement infer the format from the file. Explicit values
(``&quot;DeepLabCut&quot;``, ``&quot;SLEAP&quot;``, ``&quot;LightningPose&quot;``, ``&quot;NWB&quot;``) are
passed straight through. ``&quot;movement&quot;`` reads a netCDF file written
in the movement library&#x27;s xarray schema directly (bypassing movement&#x27;s
format-specific loaders) and validates it against the pose schema VAME
requires; the file may also include extra scalar time series with dims
``(time,)`` that ride through preprocessing.
* **video_file** (`Path or str, optional`): Path to the video file. Stored as a dataset attribute.
* **fps** (`float, optional`): Sampling rate of the video. Ignored when ``source_software`` is
``&quot;NWB&quot;`` or ``&quot;movement&quot;`` (fps is read from the file).
* **processing_module_key** (`str, optional`): Only used when ``source_software=&quot;NWB&quot;``. Name of the NWB processing
module that contains the pose estimation container. Default is
``&quot;behavior&quot;``.
* **pose_estimation_key** (`str, optional`): Only used when ``source_software=&quot;NWB&quot;``. Name of the
``ndx_pose.PoseEstimation`` object inside the processing module.
Default is ``&quot;PoseEstimation&quot;``.

**Returns**

* `xr.Dataset`: Movement-format pose estimation dataset.

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
    file_type: Optional[Literal["csv", "nwb", "slp", "h5"]] = None
) -> Tuple[pd.DataFrame, np.ndarray, xr.Dataset]
```

Read pose estimation file.

**Parameters**

* **file_path** (`str`): Path to the pose estimation file.
* **file_type** (`str, optional`): Unused; retained for backwards compatibility.

**Returns**

* `Tuple[pd.DataFrame, np.ndarray, xr.Dataset]`: Pose estimation data as a DataFrame, numpy array, and xarray Dataset.

