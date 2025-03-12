from scipy.signal import savgol_filter
import numpy as np
from pathlib import Path

from vame.logging.logger import VameLogger
from vame.io.load_poses import read_pose_estimation_file


logger_config = VameLogger(__name__)
logger = logger_config.logger


def savgol_filtering(
    config: dict,
    read_from_variable: str = "position_processed",
    save_to_variable: str = "position_processed",
) -> None:
    """
    Apply Savitzky-Golay filter to the data.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    read_from_variable : str, optional
        Variable to read from the dataset.
    save_to_variable : str, optional
        Variable to save the filtered data to.

    Returns
    -------
    None
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

        # Extract processed positions values, with shape: (time, space, keypoints, individuals)
        position = np.copy(ds[read_from_variable].values)
        filtered_position = np.copy(position)
        for individual in range(position.shape[3]):
            for keypoint in range(position.shape[2]):
                for space in range(position.shape[1]):
                    series = np.copy(position[:, space, keypoint, individual])

                    # Check if all values are zero, then skip
                    if np.all(series == 0):
                        continue

                    # Apply Savitzky-Golay filter
                    filtered_position[:, space, keypoint, individual] = savgol_filter(
                        x=series,
                        window_length=savgol_length,
                        polyorder=savgol_order,
                        axis=0,
                    )

        # Update the dataset with the filtered position values
        ds[save_to_variable] = (ds[read_from_variable].dims, filtered_position)
        ds.attrs.update({"processed_filtered": True})

        # Save the filtered dataset to file
        filtered_file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        ds.to_netcdf(
            path=filtered_file_path,
            engine="netcdf4",
        )
