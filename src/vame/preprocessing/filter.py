from scipy.signal import savgol_filter
import numpy as np
from pathlib import Path

from vame.logging.logger import VameLogger
from vame.util.data_manipulation import read_pose_estimation_file


logger_config = VameLogger(__name__)
logger = logger_config.logger


def savgol_filtering(config: dict):
    """
    Apply Savitzky-Golay filter to the data.
    """
    logger.info("Applying Savitzky-Golay filter...")
    project_path = config["project_path"]
    sessions = config["session_names"]

    savgol_length = config["savgol_length"]
    savgol_order = config["savgol_order"]
    for i, session in enumerate(sessions):
        logger.info(f"Session: {session}")
        # Read raw session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        # Extract processed positions values, with shape: (time, individuals, keypoints, space)
        position = np.copy(ds["position_processed"].values)
        filtered_position = np.copy(position)
        for individual in range(position.shape[1]):
            for keypoint in range(position.shape[2]):
                for space in range(position.shape[3]):
                    series = np.copy(position[:, individual, keypoint, space])

                    # Check if all values are zero, then skip
                    if np.all(series == 0):
                        continue

                    # Apply Savitzky-Golay filter
                    filtered_position[:, individual, keypoint, space] = savgol_filter(
                        x=series,
                        window_length=savgol_length,
                        polyorder=savgol_order,
                        axis=0,
                    )

        # Update the dataset with the filtered position values
        ds["position_processed"] = (ds["position"].dims, filtered_position)
        ds.attrs.update({"processed_filtered": True})

        # Save the filtered dataset to file
        filtered_file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        ds.to_netcdf(
            path=filtered_file_path,
            engine="scipy",
        )
