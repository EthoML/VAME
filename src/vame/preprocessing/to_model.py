import xarray as xr


def format_xarray_for_rnn(
    ds: xr.Dataset,
    read_from_variable: str = "position_processed",
):
    """
    Formats the xarray dataset for use VAME's RNN model:
    - The x and y coordinates of the centered_reference_keypoint are excluded.
    - The x coordinate of the orientation_reference_keypoint is excluded.
    - The remaining data is flattened and transposed.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset to format.
    read_from_variable : str, default="position_processed"
        The variable to read from the dataset.

    Returns
    -------
    np.ndarray
        The formatted array in the shape (n_features, n_samples).
        Where n_features = 2 * n_keypoints * n_spaces - 3.
    """
    data = ds[read_from_variable]
    centered_reference_keypoint = ds.attrs["centered_reference_keypoint"]
    orientation_reference_keypoint = ds.attrs["orientation_reference_keypoint"]

    # Get the coordinates
    individuals = data.coords["individuals"].values
    keypoints = data.coords["keypoints"].values
    spaces = data.coords["space"].values

    # Create a flattened array and infer column indices
    flattened_array = data.values.reshape(data.shape[0], -1)
    columns = [f"{ind}_{kp}_{sp}" for ind in individuals for kp in keypoints for sp in spaces]

    # Identify columns to exclude
    excluded_columns = []
    for ind in individuals:
        # Exclude both x and y for centered_reference_keypoint
        excluded_columns.append(f"{ind}_{centered_reference_keypoint}_x")
        excluded_columns.append(f"{ind}_{centered_reference_keypoint}_y")
        # Exclude only x for orientation_reference_keypoint
        excluded_columns.append(f"{ind}_{orientation_reference_keypoint}_x")

    # Filter out the excluded columns
    included_indices = [i for i, col in enumerate(columns) if col not in excluded_columns]
    filtered_array = flattened_array[:, included_indices]

    return filtered_array.T
