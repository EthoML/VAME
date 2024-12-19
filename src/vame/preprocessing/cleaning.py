from pathlib import Path
import numpy as np

from vame.logging.logger import VameLogger
from vame.util.data_manipulation import read_pose_estimation_file


logger_config = VameLogger(__name__)
logger = logger_config.logger


def lowconf_cleaning(config: dict):
    """
    Clean the low confidence data points from the dataset.
    Processes position data by:
     - setting low-confidence points to NaN
     - interpolating NaN points
    """
    project_path = config["project_path"]
    sessions = config["session_names"]
    pose_confidence = config["pose_confidence"]
    logger.info(f"Cleaning low confidence data points. Confidence threshold: {pose_confidence}")

    for i, session in enumerate(sessions):
        logger.info(f"Session: {session}")
        # Read raw session data
        file_path = str(Path(project_path) / "data" / "raw" / f"{session}.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        position = ds["position"].values
        cleaned_position = np.empty_like(position)
        confidence = ds["confidence"].values

        for individual in range(position.shape[1]):
            for keypoint in range(position.shape[2]):
                for space in range(position.shape[3]):
                    series = np.copy(position[:, individual, keypoint, space])
                    conf_series = confidence[:, individual, keypoint]

                    # Set low-confidence positions to NaN
                    nan_mask = conf_series < pose_confidence
                    series[nan_mask] = np.nan

                    # Interpolate NaN values
                    if not nan_mask.all():
                        series[nan_mask] = np.interp(
                            np.flatnonzero(nan_mask),
                            np.flatnonzero(~nan_mask),
                            series[~nan_mask],
                        )

                    # Update the position array
                    cleaned_position[:, individual, keypoint, space] = series

        # Update the dataset with the cleaned position values
        ds["position_processed"] = (ds["position"].dims, cleaned_position)
        processed_metadata = {
            "processed_confidence": True,
            "processed_egocentric": False,
            "processed_outlier": False,
            "processed_savgol": False,
        }
        ds.attrs.update(processed_metadata)

        # Save the cleaned dataset to file
        cleaned_file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        ds.to_netcdf(
            path=cleaned_file_path,
            engine="scipy",
        )
