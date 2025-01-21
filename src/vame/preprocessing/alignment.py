import numpy as np
from pathlib import Path

from vame.logging.logger import VameLogger
from vame.io.load_poses import read_pose_estimation_file


logger_config = VameLogger(__name__)
logger = logger_config.logger


def egocentrically_align_and_center(
    config: dict,
    centered_reference_keypoint: str = "snout",
    orientation_reference_keypoint: str = "tailbase",
    read_from_variable: str = "position_processed",
    save_to_variable: str = "position_egocentric_aligned",
) -> None:
    """
    Aligns the time series by first centralizing all positions around the first keypoint
    and then applying rotation to align with the line connecting the two keypoints.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    centered_reference_keypoint : str
        Name of the keypoint to use as centered reference.
    orientation_reference_keypoint : str
        Name of the keypoint to use as orientation reference.

    Returns
    -------
    None
    """
    logger.info(
        f"Egocentric alignment with references: {centered_reference_keypoint} and {orientation_reference_keypoint}"
    )
    project_path = config["project_path"]
    sessions = config["session_names"]

    for i, session in enumerate(sessions):
        logger.info(f"Session: {session}")
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        # Extract keypoint indices
        keypoints = ds.coords["keypoints"].values
        if centered_reference_keypoint not in keypoints:
            raise ValueError(
                f"Centered reference keypoint {centered_reference_keypoint} not found in dataset.",
                f"Available keypoints: {keypoints}",
            )
        if orientation_reference_keypoint not in keypoints:
            raise ValueError(
                f"Orientation reference keypoint {orientation_reference_keypoint} not found in dataset.",
                f"Available keypoints: {keypoints}",
            )
        idx1 = np.where(keypoints == centered_reference_keypoint)[0][0]
        idx2 = np.where(keypoints == orientation_reference_keypoint)[0][0]

        # Extract processed positions values, with shape: (time, individuals, keypoints, space)
        position_processed = np.copy(ds[read_from_variable].values)
        position_aligned = np.empty_like(position_processed)

        # Loop over individuals
        for individual in range(position_processed.shape[1]):
            # Shape: (time, keypoints, space)
            individual_positions = position_processed[:, individual, :, :]
            centralized_positions = np.empty_like(individual_positions)

            # Centralize all positions around the first keypoint
            for kp in range(individual_positions.shape[1]):
                for space in range(individual_positions.shape[2]):
                    centralized_positions[:, kp, space] = (
                        individual_positions[:, kp, space] - individual_positions[:, idx1, space]
                    )

            # Calculate vectors between keypoints
            vector = centralized_positions[:, idx2, :]  # Vector from keypoint1 to keypoint2
            angles = np.arctan2(vector[:, 0], vector[:, 1])  # Angles in radians

            # Apply rotation to align the second keypoint along the y-axis
            for t in range(centralized_positions.shape[0]):
                rotation_matrix = np.array(
                    [
                        [np.cos(angles[t]), -np.sin(angles[t])],
                        [np.sin(angles[t]), np.cos(angles[t])],
                    ]
                )
                frame_positions = centralized_positions[t, :, :]
                rotated_positions = (rotation_matrix @ frame_positions.T).T

                # Check and ensure the y-value of orientation_reference_keypoint is negative
                if rotated_positions[idx2, 1] > 0:
                    rotated_positions[:, :] *= -1  # Flip all coordinates

                position_aligned[t, individual, :, :] = rotated_positions

        # Update the dataset with the cleaned position values
        ds[save_to_variable] = (ds[read_from_variable].dims, position_aligned)
        ds.attrs.update(
            {
                "processed_alignment": True,
                "centered_reference_keypoint": centered_reference_keypoint,
                "orientation_reference_keypoint": orientation_reference_keypoint,
            }
        )

        # Save the aligned dataset to file
        cleaned_file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        ds.to_netcdf(
            path=cleaned_file_path,
            engine="scipy",
        )
