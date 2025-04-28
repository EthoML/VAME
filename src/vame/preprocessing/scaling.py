from pathlib import Path
import numpy as np

from vame.logging.logger import VameLogger
from vame.io.load_poses import read_pose_estimation_file

logger_config = VameLogger(__name__)
logger = logger_config.logger


def rescaling(
    config: dict,
    read_from_variable: str = "position_processed",
    save_to_variable: str = "position_scaled",
    save_logs: bool = True,
) -> None:
    """
    Rescale the position data by dividing by the individual scale values.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    read_from_variable : str, optional
        Variable to read from the dataset.
    save_to_variable : str, optional
        Variable to save the rescaled data to.
    save_logs : bool, optional
        Whether to save logs.

    Returns
    -------
    None
    """
    if save_logs:
        log_path = Path(config["project_path"]) / "logs" / "preprocessing.log"
        logger_config.add_file_handler(str(log_path))

    logger.info("Rescaling position data using individual scales...")
    project_path = config["project_path"]
    sessions = config["session_names"]

    for i, session in enumerate(sessions):
        logger.info(f"Session: {session}")
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        # Check if individual_scale exists
        if "individual_scale" not in ds:
            logger.warning(f"No individual_scale found for session {session}. Skipping rescaling.")
            continue

        # Extract position and scale values
        position = np.copy(ds[read_from_variable].values)  # shape: (time, space, keypoints, individuals)
        individual_scales = ds["individual_scale"].values  # shape: (individuals,)

        # Create scaled position array
        scaled_position = np.empty_like(position)

        # Apply scaling to each individual
        for individual in range(position.shape[3]):
            scale = individual_scales[individual]

            # Handle potential division by zero or very small values
            if scale < 1e-10:
                logger.warning(f"Very small scale ({scale}) for individual {individual}. Using default scale of 1.0")
                scale = 1.0

            # Scale all positions for this individual
            scaled_position[:, :, :, individual] = position[:, :, :, individual] / scale

        # Update the dataset with the scaled position values
        ds[save_to_variable] = (ds[read_from_variable].dims, scaled_position)
        ds.attrs.update({"processed_rescaled": "True"})

        # Save the updated dataset to file
        scaled_file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        ds.to_netcdf(
            path=scaled_file_path,
            engine="netcdf4",
        )
