from pathlib import Path
import numpy as np
from scipy.stats import iqr

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

        perc_interp_points = np.zeros((position.shape[1], position.shape[2]))
        for individual in range(position.shape[1]):
            for keypoint in range(position.shape[2]):
                conf_series = confidence[:, individual, keypoint]
                nan_mask = conf_series < pose_confidence
                perc_interp_points[individual, keypoint] = 100 * np.sum(nan_mask) / len(nan_mask)
                for space in range(position.shape[3]):
                    # Set low-confidence positions to NaN
                    series = np.copy(position[:, individual, keypoint, space])
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
        ds.attrs.update({"processed_confidence": True})

        ds["percentage_low_confidence"] = (["individual", "keypoint"], perc_interp_points)

        # Save the cleaned dataset to file
        cleaned_file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        ds.to_netcdf(
            path=cleaned_file_path,
            engine="scipy",
        )


def outlier_cleaning(config: dict):
    """
    Clean the outliers from the dataset.
    Processes position data by:
     - setting outlier points to NaN
     - interpolating NaN points
    """
    logger.info("Cleaning outliers with Z-score transformation and IQR cutoff.")
    project_path = config["project_path"]
    sessions = config["session_names"]

    for i, session in enumerate(sessions):
        logger.info(f"Session: {session}")
        # Read raw session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        position = np.copy(ds["position_processed"].values)
        cleaned_position = np.copy(position)

        perc_interp_points = np.zeros((position.shape[1], position.shape[2], position.shape[3]))

        for individual in range(position.shape[1]):
            for keypoint in range(position.shape[2]):
                for space in range(position.shape[3]):
                    series = np.copy(position[:, individual, keypoint, space])

                    # Check if all values are zero, then skip
                    if np.all(series == 0):
                        continue

                    # Calculate Z-score
                    z_series = (series - np.nanmean(series)) / np.nanstd(series)

                    # Set outlier positions to NaN, based on IQR cutoff
                    if config["robust"]:
                        iqr_factor = config["iqr_factor"]
                        iqr_val = iqr(z_series)
                        outlier_mask = np.abs(z_series) > iqr_factor * iqr_val
                        z_series[outlier_mask] = np.nan
                        perc_interp_points[individual, keypoint, space] = 100 * np.sum(outlier_mask) / len(outlier_mask)

                        # Interpolate NaN values
                        if not outlier_mask.all():
                            z_series[outlier_mask] = np.interp(
                                np.flatnonzero(outlier_mask),
                                np.flatnonzero(~outlier_mask),
                                z_series[~outlier_mask],
                            )

                        # Redo the z-score to remove the bias of the now-removed outliers
                        z_series = (z_series - np.nanmean(z_series)) / np.nanstd(z_series)

                    # Update the processed position array
                    cleaned_position[:, individual, keypoint, space] = z_series

        # Update the dataset with the cleaned position values
        ds["position_processed"] = (ds["position"].dims, cleaned_position)
        ds.attrs.update({"processed_outliers": True})

        ds["percentage_iqr_outliers"] = (["individual", "keypoint", "space"], perc_interp_points)

        # Save the cleaned dataset to file
        cleaned_file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        ds.to_netcdf(
            path=cleaned_file_path,
            engine="scipy",
        )
