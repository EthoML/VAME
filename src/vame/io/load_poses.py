from typing import Literal, Optional, Tuple
from pathlib import Path
from movement.io import load_poses as mio_load_poses
import xarray as xr
import numpy as np
import pandas as pd


def load_pose_estimation(
    pose_estimation_file: Path | str,
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose", "NWB"],
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
        Path to the pose estimation file.
    video_file : Path or str
        Path to the video file.
    fps : float, optional
        Sampling rate of the video.
    source_software : Literal["DeepLabCut", "SLEAP", "LightningPose", "NWB"]
        Source software used for pose estimation. Use "NWB" to read an
        ``ndx-pose`` PoseEstimation from an NWB file.
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
    ds : xarray.Dataset
        Pose estimation dataset.
    """
    if source_software == "NWB":
        ds = mio_load_poses.from_nwb_file(
            file=pose_estimation_file,
            processing_module_key=processing_module_key,
            pose_estimation_key=pose_estimation_key,
        )
    else:
        ds = mio_load_poses.from_file(
            file_path=pose_estimation_file,
            source_software=source_software,
            fps=fps,
        )
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
        ds_in_memory = tmp_ds.load()  # read entire file into memory
    return ds_in_memory


# def load_vame_dataset_lock(ds_path: Path | str) -> xr.Dataset:
#     """
#     Load VAME dataset with file locking to prevent conflicts.

#     Parameters
#     ----------
#     ds_path : Path or str
#         Path to the netCDF dataset.

#     Returns
#     -------
#     xr.Dataset
#         VAME dataset loaded into memory.
#     """
#     import portalocker

#     ds_path = Path(ds_path)
#     lock_path = ds_path.parent / f"{ds_path.name}.lock"

#     # Use portalocker.Lock which supports the `timeout` keyword.
#     with portalocker.Lock(str(lock_path), mode="w", flags=portalocker.LOCK_SH) as _:
#         with xr.open_dataset(ds_path, engine="netcdf4") as tmp_ds:
#             ds_in_memory = tmp_ds.load()  # Load the dataset into memory.
#     return ds_in_memory


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
    file_type : PoseEstimationFiletype
        Type of the pose estimation file. Supported types are 'csv' and 'nwb'.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Tuple containing the pose estimation data as a pandas DataFrame and a numpy array.
    """
    ds = load_vame_dataset(ds_path=file_path)
    data = nc_to_dataframe(ds)
    data_mat = pd.DataFrame.to_numpy(data)
    return data, data_mat, ds
