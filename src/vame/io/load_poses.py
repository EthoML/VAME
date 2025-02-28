from typing import Literal, Optional, Tuple
from pathlib import Path
from movement.io import load_poses as mio_load_poses
import xarray as xr
import numpy as np
import pandas as pd


def load_pose_estimation(
    pose_estimation_file: Path | str,
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
    video_file: Optional[Path | str] = None,
    fps: Optional[float] = None,
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
    source_software : Literal["DeepLabCut", "SLEAP", "LightningPose"]
        Source software used for pose estimation.

    Returns
    -------
    ds : xarray.Dataset
        Pose estimation dataset.
    """
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
    # Windows will not allow opened files to be overwritten,
    # so we need to load data into memory, close the file and move on with the operations
    with xr.open_dataset(ds_path, engine="scipy") as tmp_ds:
        ds_in_memory = tmp_ds.load()  # read entire file into memory
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
    path_to_pose_nwb_series_data: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, xr.Dataset]:
    """
    Read pose estimation file.

    Parameters
    ----------
    file_path : str
        Path to the pose estimation file.
    file_type : PoseEstimationFiletype
        Type of the pose estimation file. Supported types are 'csv' and 'nwb'.
    path_to_pose_nwb_series_data : str, optional
        Path to the pose data inside the nwb file, by default None

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Tuple containing the pose estimation data as a pandas DataFrame and a numpy array.
    """
    ds = load_vame_dataset(ds_path=file_path)
    data = nc_to_dataframe(ds)
    data_mat = pd.DataFrame.to_numpy(data)
    return data, data_mat, ds
    # if file_type == PoseEstimationFiletype.csv:
    #     data = pd.read_csv(file_path, skiprows=2, index_col=0)
    #     if "coords" in data:
    #         data = data.drop(columns=["coords"], axis=1)
    #     data_mat = pd.DataFrame.to_numpy(data)
    #     return data, data_mat
    # elif file_type == PoseEstimationFiletype.nwb:
    #     if not path_to_pose_nwb_series_data:
    #         raise ValueError("Path to pose nwb series data is required.")
    #     data = get_dataframe_from_pose_nwb_file(
    #         file_path=file_path,
    #         path_to_pose_nwb_series_data=path_to_pose_nwb_series_data,
    #     )
    #     data_mat = pd.DataFrame.to_numpy(data)
    #     return data, data_mat
    # raise ValueError(f"Filetype {file_type} not supported")
