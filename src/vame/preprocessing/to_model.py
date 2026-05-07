import xarray as xr
import numpy as np
from typing import Tuple, Dict, Any


def format_xarray_for_rnn(
    ds: xr.Dataset,
    read_from_variable: str = "position_processed",
    keypoints: list[str] | None = None,
    extra_features: list[str] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Formats the xarray dataset for use VAME's RNN model:
    - The x and y coordinates of the centered_reference_keypoint are excluded.
    - The x coordinate of the orientation_reference_keypoint is excluded.
    - The remaining data is flattened and transposed.
    - Optionally, pre-computed scalar features (speed, head direction, ...) are
      appended as additional rows after the pose-derived features.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset to format.
    read_from_variable : str, default="position_processed"
        The variable to read from the dataset.
    keypoints : list[str] | None, optional
        A list of keypoints to include in the output. If None, all keypoints are
        included. If provided, only the specified keypoints will be included in the output.
    extra_features : list[str] | None, optional
        Names of pre-computed scalar features to append after the pose-derived
        features. Each must exist as a data variable in ``ds`` with dims
        ``("time",)`` or ``("time", "individuals")``. VAME applies no
        preprocessing to these — the user is responsible for alignment, scaling,
        and NaN handling. See ``vame.validate_extra_features``.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        A tuple containing:
        - The formatted array in the shape (n_features, n_samples)
        - A dictionary with feature provenance and processing information
        Where n_features = 2 * n_keypoints * n_spaces - 3 + len(extra_features or []).
    """
    data = ds[read_from_variable]
    centered_reference_keypoint = ds.attrs["centered_reference_keypoint"]
    orientation_reference_keypoint = ds.attrs["orientation_reference_keypoint"]

    # Select the first individual
    individuals = data.coords["individuals"].values
    first_individual = individuals[0]
    data = data.sel(individuals=first_individual)

    # Extract spaces and keypoints from the dataset
    spaces = data.coords["space"].values
    original_keypoints = data.coords["keypoints"].values
    if keypoints is None:
        keypoints = list(original_keypoints)
    else:
        data = data.sel(keypoints=keypoints)

    # Create an array with filtered data (n_samples, n_features - 3)
    filtered_array = []
    feature_mapping = []
    excluded_features = []
    feature_index = 0

    for kp in keypoints:
        if kp == centered_reference_keypoint:
            # Track excluded features
            for sp in spaces:
                excluded_features.append(f"{kp}_{sp}")
            continue
        for sp in spaces:
            if sp == "x" and kp == orientation_reference_keypoint:
                # Track excluded feature
                excluded_features.append(f"{kp}_{sp}")
                continue
            column_data = data.sel(keypoints=kp, space=sp).values.reshape(-1)
            filtered_array.append(column_data)

            # Track feature mapping for metadata
            feature_mapping.append({
                "index": feature_index,
                "keypoint": kp,
                "coordinate": sp,
                "feature_name": f"{kp}_{sp}"
            })
            feature_index += 1

    # Append pre-computed extra (non-pose) features. Each contributes one row,
    # read from individuals[0] to match the pose-feature handling above.
    extra_names = list(extra_features or [])
    for name in extra_names:
        if name not in ds.data_vars:
            raise ValueError(
                f"Extra feature '{name}' not found in dataset; "
                f"available data variables: {list(ds.data_vars)}"
            )
        da = ds[name]
        if "individuals" in da.dims:
            series = da.sel(individuals=first_individual).values
        else:
            series = da.values
        series = np.asarray(series).reshape(-1)
        filtered_array.append(series)
        feature_mapping.append({
            "index": feature_index,
            "extra_feature": name,
            "feature_name": name,
        })
        feature_index += 1

    filtered_array = np.array(filtered_array)

    # Create metadata dictionary
    metadata = {
        "feature_mapping": feature_mapping,
        "parameters": {
            "read_from_variable": read_from_variable,
            "keypoints_used": list(keypoints),
            "keypoints_available": list(original_keypoints),
            "centered_reference_keypoint": centered_reference_keypoint,
            "orientation_reference_keypoint": orientation_reference_keypoint,
            "excluded_features": excluded_features,
            "extra_features": extra_names,
            "total_features": len(feature_mapping),
            "data_shape": filtered_array.shape,
        }
    }

    return filtered_array, metadata
