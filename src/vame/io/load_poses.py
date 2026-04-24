from typing import Literal, Optional, Tuple
from pathlib import Path
from movement.io.load import load_dataset
import xarray as xr
import numpy as np
import pandas as pd


def load_pose_estimation(
    pose_estimation_file: Path | str,
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose", "NWB", "auto"] = "auto",
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
        passed straight through.
    video_file : Path or str, optional
        Path to the video file. Stored as a dataset attribute.
    fps : float, optional
        Sampling rate of the video. Ignored when ``source_software`` is
        ``"NWB"`` (fps is read from the file).
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
