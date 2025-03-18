from pathlib import Path
import numpy as np
from scipy.stats import iqr

from vame.logging.logger import VameLogger
from vame.io.load_poses import read_pose_estimation_file


logger_config = VameLogger(__name__)
logger = logger_config.logger


def lowconf_cleaning(
    config: dict,
    read_from_variable: str = "position_processed",
    save_to_variable: str = "position_processed",
) -> None:
    """
    Clean the low confidence data points from the dataset. Processes position data by:
     - setting low-confidence points to NaN
     - interpolating NaN points

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    read_from_variable : str, optional
        Variable to read from the dataset.
    save_to_variable : str, optional
        Variable to save the cleaned data to.

    Returns
    -------
    None
    """
    project_path = config["project_path"]
    sessions = config["session_names"]
    pose_confidence = config["pose_confidence"]
    logger.info(f"Cleaning low confidence data points. Confidence threshold: {pose_confidence}")

    for i, session in enumerate(sessions):
        logger.info(f"Session: {session}")
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        position = ds[read_from_variable].values  # shape: (time, space, keypoints, individuals)
        cleaned_position = np.empty_like(position)
        confidence = ds["confidence"].values  # shape: (time, keypoints, individuals)

        # Initialize percentage array with dimensions matching position order (space, keypoints, individuals)
        perc_interp_points = np.zeros((position.shape[1], position.shape[2], position.shape[3]))
        for individual in range(position.shape[3]):
            for keypoint in range(position.shape[2]):
                # Get confidence for this keypoint and individual
                conf_series = confidence[:, keypoint, individual].copy()
                for space in range(position.shape[1]):
                    # Set low-confidence positions to NaN
                    nan_mask = conf_series < pose_confidence
                    series = np.copy(position[:, space, keypoint, individual])
                    series[nan_mask] = np.nan

                    # Update nan_mask because the series might come with NaN values previously
                    nan_mask = np.isnan(series)

                    perc_interp_points[space, keypoint, individual] = 100 * np.sum(nan_mask) / len(nan_mask)

                    # Interpolate NaN values
                    if not nan_mask.all():
                        series[nan_mask] = np.interp(
                            np.flatnonzero(nan_mask),
                            np.flatnonzero(~nan_mask),
                            series[~nan_mask],
                        )

                    # Update the position array
                    cleaned_position[:, space, keypoint, individual] = series

        # Update the dataset with the cleaned position values
        ds[save_to_variable] = (ds[read_from_variable].dims, cleaned_position)
        ds.attrs.update({"processed_confidence": "True"})

        ds["percentage_low_confidence"] = (["space", "keypoints", "individuals"], perc_interp_points)

        # Save the cleaned dataset to file
        cleaned_file_path = Path(project_path) / "data" / "processed" / f"{session}_processed.nc"
        ds.to_netcdf(
            path=cleaned_file_path,
            engine="netcdf4",
        )


def outlier_cleaning(
    config: dict,
    read_from_variable: str = "position_processed",
    save_to_variable: str = "position_processed",
) -> None:
    """
    Clean the outliers from the dataset. Processes position data by:
     - rescaling data by individual anatomical scale
     - setting outlier points to NaN based on IQR cutoff
     - interpolating NaN points

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    read_from_variable : str, optional
        Variable to read from the dataset.
    save_to_variable : str, optional
        Variable to save the cleaned data to.

    Returns
    -------
    None
    """
    logger.info("Cleaning outliers with anatomical rescaling and IQR cutoff.")
    project_path = config["project_path"]
    sessions = config["session_names"]

    for i, session in enumerate(sessions):
        logger.info(f"Session: {session}")
        # Read raw session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        position = np.copy(ds[read_from_variable].values)  # shape: (time, space, keypoints, individuals)
        cleaned_position = np.copy(position)

        # Initialize percentage array with dimensions matching position order (space, keypoints, individuals)
        perc_interp_points = np.zeros((position.shape[1], position.shape[2], position.shape[3]))

        for individual in range(position.shape[3]):
            for keypoint in range(position.shape[2]):
                for space in range(position.shape[1]):
                    series = np.copy(position[:, space, keypoint, individual])

                    # Check if all values are zero, then skip
                    if np.all(series == 0):
                        continue

                    # Get individual scale for this individual
                    individual_scale = ds["individual_scale"].values[individual]

                    # Rescale by individual_scale
                    rescaled_series = series / individual_scale

                    # Set outlier positions to NaN, based on IQR cutoff
                    if config["robust"]:
                        iqr_factor = config["iqr_factor"]
                        iqr_val = iqr(rescaled_series)
                        # Calculate median for rescaled series
                        median_val = np.nanmedian(rescaled_series)
                        # Define bounds using median and IQR
                        lower_bound = median_val - iqr_factor * iqr_val
                        upper_bound = median_val + iqr_factor * iqr_val
                        # Identify outliers
                        outlier_mask = (rescaled_series < lower_bound) | (rescaled_series > upper_bound)

                        # Set outliers to NaN
                        rescaled_series[outlier_mask] = np.nan
                        perc_interp_points[space, keypoint, individual] = (
                            100 * np.sum(outlier_mask) / len(outlier_mask)
                        )

                        # Interpolate NaN values
                        if not outlier_mask.all():
                            rescaled_series[outlier_mask] = np.interp(
                                np.flatnonzero(outlier_mask),
                                np.flatnonzero(~outlier_mask),
                                rescaled_series[~outlier_mask],
                            )

                        # CHANGED WHEN REMOVED Z-SCORE
                        # # Redo the z-score to remove the bias of the now-removed outliers
                        # z_series = (z_series - np.nanmean(z_series)) / np.nanstd(z_series)

                    # Store the rescaled series
                    cleaned_position[:, space, keypoint, individual] = rescaled_series

        # Update the dataset with the cleaned position values
        ds[save_to_variable] = (ds[read_from_variable].dims, cleaned_position)
        ds.attrs.update({
            "processed_outliers": "True",
            "rescaled_by_individual_scale": "True"
        })

        ds["percentage_iqr_outliers"] = (["space", "keypoints", "individuals"], perc_interp_points)

        # Save the cleaned dataset to file
        cleaned_file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        ds.to_netcdf(
            path=cleaned_file_path,
            engine="netcdf4",
        )
