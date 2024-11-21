from typing import Literal
from pathlib import Path
from movement.io import load_poses as mio_load_poses
import xarray as xr


def load_pose_estimation(
    pose_estimation_file: Path | str,
    video_file: Path | str,
    fps: int,
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
) -> xr.Dataset:
    """
    Load pose estimation data.

    Parameters:
    -----------
    pose_estimation_file : Path or str
        Path to the pose estimation file.
    video_file : Path or str
        Path to the video file.
    fps : int
        Sampling rate of the video.
    source_software : Literal["DeepLabCut", "SLEAP", "LightningPose"]
        Source software used for pose estimation.

    Returns:
    --------
    ds : xarray.Dataset
        Pose estimation dataset.
    """
    ds = mio_load_poses.from_file(
        file_path=pose_estimation_file,
        source_software=source_software,
        fps=fps,
    )
    ds.attrs["video_path"] = str(video_file)
    return ds


def load_vame_dataset(ds_path: Path | str) -> xr.Dataset:
    """
    Load VAME dataset.

    Parameters:
    -----------
    ds_path : Path or str
        Path to the netCDF dataset.

    Returns:
    --------
    """
    return xr.open_dataset(ds_path, engine="scipy")
