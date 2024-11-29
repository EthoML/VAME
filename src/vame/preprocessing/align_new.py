import numpy as np
import pandas as pd
import xarray as xr


def align_time_series(data, keypoint1, keypoint2, confidence_threshold):
    """
    Aligns the time series by first centralizing all positions around the first keypoint
    and then applying rotation to align with the line connecting the two keypoints.
    Handles low-confidence points by replacing them with NaNs and interpolating.

    Parameters:
    - data (xarray.Dataset): The input dataset.
    - keypoint1 (str): The name of the first reference keypoint.
    - keypoint2 (str): The name of the second reference keypoint.
    - confidence_threshold (float): Confidence threshold below which points are replaced with NaNs.

    Returns:
    - xarray.Dataset: The dataset with a new 'position_aligned' variable.
    """
    # Extract keypoint indices
    keypoints = data.coords["keypoints"].values
    idx1 = np.where(keypoints == keypoint1)[0][0]
    idx2 = np.where(keypoints == keypoint2)[0][0]

    # Extract positions and confidence values
    positions = data["position"].values  # Shape: (time, individuals, keypoints, space)
    confidence = data["confidence"].values  # Shape: (time, individuals, keypoints)

    aligned_positions = np.empty_like(positions)  # Preallocate aligned positions

    # Loop over individuals
    for ind in range(positions.shape[1]):
        individual_positions = positions[:, ind, :, :]  # Shape: (time, keypoints, space)
        individual_confidence = confidence[:, ind, :]  # Shape: (time, keypoints)

        # Replace low-confidence points with NaN
        for kp in range(individual_positions.shape[1]):  # Loop over keypoints
            for dim in range(2):  # Loop over x and y
                low_confidence = individual_confidence[:, kp] < confidence_threshold
                individual_positions[low_confidence, kp, dim] = np.nan

        # Interpolate NaN values
        for kp in range(individual_positions.shape[1]):  # Loop over keypoints
            for dim in range(2):  # Loop over x and y
                series = pd.Series(individual_positions[:, kp, dim])
                individual_positions[:, kp, dim] = (
                    series.interpolate(method="linear", limit_direction="both")
                    .bfill()  # Backward fill for initial NaNs
                    .ffill()  # Forward fill for final NaNs
                    .values
                )

        # Centralize all positions around the first keypoint
        centralized_positions = individual_positions - individual_positions[:, idx1, :][:, np.newaxis, :]

        # Calculate vectors between keypoints
        vector = centralized_positions[:, idx2, :]  # Vector from keypoint1 to keypoint2
        angles = np.arctan2(vector[:, 1], vector[:, 0])  # Angles in radians

        # Apply rotation to align the second keypoint along the x-axis
        for t in range(centralized_positions.shape[0]):
            rotation_matrix = np.array(
                [
                    [np.cos(-angles[t]), -np.sin(-angles[t])],
                    [np.sin(-angles[t]), np.cos(-angles[t])],
                ]
            )
            frame_positions = centralized_positions[t, :, :]
            rotated_positions = (rotation_matrix @ frame_positions.T).T
            aligned_positions[t, ind, :, :] = rotated_positions

    # Add new variable to the dataset
    data["position_aligned"] = (
        ("time", "individuals", "keypoints", "space"),
        aligned_positions,
    )

    return data
