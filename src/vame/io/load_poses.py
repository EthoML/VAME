from typing import Literal, Optional, Tuple
from pathlib import Path
from movement.io.load import load_dataset
import xarray as xr
import numpy as np
import pandas as pd


def _validate_movement_schema(ds: xr.Dataset) -> None:
    """
    Validate that a Dataset matches the movement-format pose schema VAME expects.

    Aggregates all problems into a single ``ValueError`` so users see the full
    set of issues at once instead of fixing them one at a time.
    """
    expected_position_dims = ("time", "space", "keypoints", "individuals")
    expected_confidence_dims = ("time", "keypoints", "individuals")
    problems: list[str] = []

    if "position" not in ds.data_vars:
        problems.append("missing required data variable 'position'")
    if "confidence" not in ds.data_vars:
        problems.append("missing required data variable 'confidence'")

    if "position" in ds.data_vars:
        position = ds["position"]
        if not np.issubdtype(position.dtype, np.number):
            problems.append(f"'position' has non-numeric dtype {position.dtype}")
        if position.dims != expected_position_dims:
            problems.append(
                f"'position' has dims {position.dims}; expected {expected_position_dims}"
            )

    if "confidence" in ds.data_vars:
        confidence = ds["confidence"]
        if not np.issubdtype(confidence.dtype, np.number):
            problems.append(f"'confidence' has non-numeric dtype {confidence.dtype}")
        if confidence.dims != expected_confidence_dims:
            problems.append(
                f"'confidence' has dims {confidence.dims}; expected {expected_confidence_dims}"
            )

    if "time" not in ds.coords:
        problems.append("missing required coord 'time'")
    elif ds.sizes.get("time", 0) == 0:
        problems.append("'time' coord is empty")

    if "space" not in ds.coords:
        problems.append("missing required coord 'space'")
    else:
        space_values = list(ds["space"].values)
        if space_values != ["x", "y"]:
            problems.append(f"'space' coord values are {space_values}; expected ['x', 'y']")

    if "keypoints" not in ds.coords:
        problems.append("missing required coord 'keypoints'")
    elif ds.sizes.get("keypoints", 0) == 0:
        problems.append("'keypoints' coord is empty")

    if "individuals" not in ds.coords:
        problems.append("missing required coord 'individuals'")
    elif ds.sizes.get("individuals", 0) == 0:
        problems.append("'individuals' coord is empty")

    if "position" in ds.data_vars and "confidence" in ds.data_vars:
        for dim in ("time", "keypoints", "individuals"):
            p_size = ds["position"].sizes.get(dim)
            c_size = ds["confidence"].sizes.get(dim)
            if p_size is not None and c_size is not None and p_size != c_size:
                problems.append(
                    f"size mismatch on '{dim}': position={p_size}, confidence={c_size}"
                )

    for name, da in ds.data_vars.items():
        if name in ("position", "confidence"):
            continue
        if da.dims != ("time",):
            problems.append(
                f"unsupported data variable '{name}' with dims {da.dims}; "
                "only (time,) extras are accepted alongside position/confidence"
            )

    if problems:
        raise ValueError(
            "movement-format schema validation failed with "
            f"{len(problems)} problem(s):\n  - " + "\n  - ".join(problems)
        )


def load_pose_estimation(
    pose_estimation_file: Path | str,
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose", "NWB", "auto", "movement"] = "auto",
    video_file: Optional[Path | str] = None,
    fps: Optional[float] = None,
    processing_module_key: str = "behavior",
    pose_estimation_key: str = "PoseEstimation",
) -> xr.Dataset:
    """
    Load pose estimation data.

    Parameters
    ----------
    pose_estimation_file : Path or str
        Path to the pose estimation file. Dispatched through movement's
        unified loader, which auto-detects format from extension and contents.
    source_software : str, optional
        Source software used for pose estimation. Defaults to ``"auto"``, which
        lets movement infer the format from the file. Explicit values
        (``"DeepLabCut"``, ``"SLEAP"``, ``"LightningPose"``, ``"NWB"``) are
        passed straight through. ``"movement"`` reads a netCDF file written
        in the movement library's xarray schema directly (bypassing movement's
        format-specific loaders) and validates it against the pose schema VAME
        requires; the file may also include extra scalar time series with dims
        ``(time,)`` that ride through preprocessing.
    video_file : Path or str, optional
        Path to the video file. Stored as a dataset attribute.
    fps : float, optional
        Sampling rate of the video. Ignored when ``source_software`` is
        ``"NWB"`` or ``"movement"`` (fps is read from the file).
    processing_module_key : str, optional
        Only used when ``source_software="NWB"``. Name of the NWB processing
        module that contains the pose estimation container. Default is
        ``"behavior"``.
    pose_estimation_key : str, optional
        Only used when ``source_software="NWB"``. Name of the
        ``ndx_pose.PoseEstimation`` object inside the processing module.
        Default is ``"PoseEstimation"``.

    Returns
    -------
    xr.Dataset
        Movement-format pose estimation dataset.
    """
    file_path = Path(pose_estimation_file)

    if source_software == "movement":
        ds = load_vame_dataset(ds_path=file_path)
        _validate_movement_schema(ds)
        if video_file:
            ds.attrs["video_path"] = str(video_file)
        return ds

    nwb_kwargs = {}
    if source_software == "NWB":
        nwb_kwargs = {
            "processing_module_key": processing_module_key,
            "pose_estimation_key": pose_estimation_key,
        }
    ds = load_dataset(file=file_path, source_software=source_software, fps=fps, **nwb_kwargs)
    # movement's NWB loader stores attrs as Path objects, which xarray's
    # netCDF writer rejects. Coerce to str so downstream ds.to_netcdf works.
    ds.attrs = {k: (str(v) if isinstance(v, Path) else v) for k, v in ds.attrs.items()}

    if video_file:
        ds.attrs["video_path"] = str(video_file)
    return ds


def load_vame_dataset(ds_path: Path | str) -> xr.Dataset:
    """
    Load VAME dataset.

    Parameters
    ----------
    ds_path : Path or str
        Path to the netCDF dataset.

    Returns
    -------
    xr.Dataset
        VAME dataset
    """
    with xr.open_dataset(ds_path, engine="netcdf4") as tmp_ds:
        ds_in_memory = tmp_ds.load()
    return ds_in_memory


def nc_to_dataframe(nc_data):
    keypoints = nc_data["keypoints"].values
    space = nc_data["space"].values

    # Flatten position data
    position_data = nc_data["position"].isel(individuals=0).values
    position_column_names = [f"{keypoint}_{sp}" for keypoint in keypoints for sp in space]
    position_flattened = position_data.reshape(position_data.shape[0], -1)

    # Create a DataFrame for position data
    position_df = pd.DataFrame(position_flattened, columns=position_column_names)

    # Extract and flatten confidence data
    confidence_data = nc_data["confidence"].isel(individuals=0).values
    confidence_column_names = [f"{keypoint}_confidence" for keypoint in keypoints]
    confidence_flattened = confidence_data.reshape(confidence_data.shape[0], -1)
    confidence_df = pd.DataFrame(confidence_flattened, columns=confidence_column_names)

    # Combine position and confidence data
    combined_df = pd.concat([position_df, confidence_df], axis=1)

    # Reorder columns: keypoint_x, keypoint_y, keypoint_confidence
    reordered_columns = []
    for keypoint in keypoints:
        reordered_columns.extend([f"{keypoint}_x", f"{keypoint}_y", f"{keypoint}_confidence"])

    combined_df = combined_df[reordered_columns]

    return combined_df


def read_pose_estimation_file(
    file_path: str,
    file_type: Optional[Literal["csv", "nwb", "slp", "h5"]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, xr.Dataset]:
    """
    Read pose estimation file.

    Parameters
    ----------
    file_path : str
        Path to the pose estimation file.
    file_type : str, optional
        Unused; retained for backwards compatibility.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, xr.Dataset]
        Pose estimation data as a DataFrame, numpy array, and xarray Dataset.
    """
    ds = load_vame_dataset(ds_path=file_path)
    data = nc_to_dataframe(ds)
    data_mat = pd.DataFrame.to_numpy(data)
    return data, data_mat, ds
