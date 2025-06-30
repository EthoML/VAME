import xarray as xr
import numpy as np


def format_xarray_for_rnn(
    ds: xr.Dataset,
    read_from_variable: str = "position_processed",
    keypoints: list[str] | None = None,
) -> np.ndarray:
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
    keypoints : list[str] | None, optional
        A list of keypoints to include in the output. If None, all keypoints are
        included. If provided, only the specified keypoints will be included in the output.

    Returns
    -------
    np.ndarray
        The formatted array in the shape (n_features, n_samples).
        Where n_features = 2 * n_keypoints * n_spaces - 3.
    """
    data = ds[read_from_variable]
    centered_reference_keypoint = ds.attrs["centered_reference_keypoint"]
    orientation_reference_keypoint = ds.attrs["orientation_reference_keypoint"]

    # Select the first individual
    individuals = data.coords["individuals"].values
    data = data.sel(individuals=individuals[0])

    # Extract spaces and keypoints from the dataset
    spaces = data.coords["space"].values
    if keypoints is None:
        keypoints = data.coords["keypoints"].values
    else:
        data = data.sel(keypoints=keypoints)

    # Create an array with filtered data (n_samples, n_features - 3)
    filtered_array = []
    for kp in keypoints:
        if kp == centered_reference_keypoint:
            continue
        for sp in spaces:
            if sp == "x" and kp == orientation_reference_keypoint:
                continue
            column_data = data.sel(keypoints=kp, space=sp).values.reshape(-1)
            filtered_array.append(column_data)
    filtered_array = np.array(filtered_array).T

    return filtered_array
