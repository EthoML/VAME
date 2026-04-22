import itertools
from typing import Literal, Tuple, Dict, Any

import xarray as xr
import numpy as np

# Variables that are part of the standard VAME/movement schema and should not
# be treated as extra features when extra_features="all".
_STANDARD_VARIABLES = {"position", "confidence"}


def _resolve_extra_features(
    ds: xr.Dataset,
    extra_features: list[str] | Literal["all"] | None,
) -> list[str]:
    """Return the resolved list of extra feature variable names."""
    if extra_features is None:
        return []
    if extra_features == "all":
        return [v for v in ds.data_vars if v not in _STANDARD_VARIABLES and not v.startswith("position_")]
    return list(extra_features)


def _build_dim_labels(var: xr.DataArray, non_time_dims: list[str]) -> list[str]:
    """
    Build a flat list of string labels for every combination of coordinate values
    across the non-time (and non-individuals) dimensions of var.

    For (space, keypoints) this gives ["x_snout", "x_tailbase", "y_snout", …].
    For a single integer axis it falls back to "0", "1", …
    """
    if not non_time_dims:
        return ["scalar"]

    coord_values = []
    for dim in non_time_dims:
        if dim in var.coords:
            coord_values.append([str(v) for v in var.coords[dim].values])
        else:
            coord_values.append([str(i) for i in range(var.sizes[dim])])

    return ["_".join(combo) for combo in itertools.product(*coord_values)]


def format_xarray_for_rnn(
    ds: xr.Dataset,
    read_from_variable: str = "position_processed",
    keypoints: list[str] | None = None,
    extra_features: list[str] | Literal["all"] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Formats the xarray dataset for use in VAME's RNN model.

    - The x and y coordinates of the centered_reference_keypoint are excluded.
    - The x coordinate of the orientation_reference_keypoint is excluded.
    - The remaining keypoint data is flattened and stacked.
    - Any variables listed in extra_features are appended after the keypoint features.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset to format.
    read_from_variable : str, default="position_processed"
        The variable to read from the dataset.
    keypoints : list[str] | None, optional
        Keypoints to include. If None, all keypoints are included.
    extra_features : list[str] | "all" | None, optional
        Extra data variables to append as features. Pass a list of variable names,
        ``"all"`` to include every non-standard variable in the dataset, or ``None``
        to use no extra features.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - Array of shape (n_features, n_samples)
        - Metadata dictionary with feature provenance information
    """
    data = ds[read_from_variable]
    centered_reference_keypoint = ds.attrs["centered_reference_keypoint"]
    orientation_reference_keypoint = ds.attrs["orientation_reference_keypoint"]

    # Select the first individual
    individuals = data.coords["individuals"].values
    data = data.sel(individuals=individuals[0])

    # Extract spaces and keypoints from the dataset
    spaces = data.coords["space"].values
    original_keypoints = data.coords["keypoints"].values
    if keypoints is None:
        keypoints = list(original_keypoints)
    else:
        data = data.sel(keypoints=keypoints)

    filtered_array = []
    feature_mapping = []
    excluded_features = []
    feature_index = 0

    for kp in keypoints:
        if kp == centered_reference_keypoint:
            for sp in spaces:
                excluded_features.append(f"{kp}_{sp}")
            continue
        for sp in spaces:
            if sp == "x" and kp == orientation_reference_keypoint:
                excluded_features.append(f"{kp}_{sp}")
                continue
            column_data = data.sel(keypoints=kp, space=sp).values.reshape(-1)
            filtered_array.append(column_data)
            feature_mapping.append({
                "index": feature_index,
                "source": "keypoint",
                "keypoint": kp,
                "coordinate": sp,
                "feature_name": f"{kp}_{sp}",
            })
            feature_index += 1

    # Append extra feature variables
    resolved_extra = _resolve_extra_features(ds, extra_features)
    for var_name in resolved_extra:
        if var_name not in ds:
            raise ValueError(
                f"Extra feature '{var_name}' not found in dataset. "
                f"Available variables: {list(ds.data_vars)}"
            )
        var = ds[var_name]

        if "individuals" in var.dims:
            var = var.sel(individuals=individuals[0])

        arr = var.values
        non_time_dims = [d for d in var.dims if d != "time"]

        if arr.ndim == 1:
            # Shape: (time,)
            rows = [arr]
            labels = ["scalar"]
        else:
            # Move time to the last axis, then flatten everything else
            time_axis = var.dims.index("time")
            axes = list(range(arr.ndim))
            axes.remove(time_axis)
            axes.append(time_axis)
            arr = arr.transpose(axes)
            n_time = arr.shape[-1]
            rows = arr.reshape(-1, n_time)
            labels = _build_dim_labels(var, non_time_dims)

        for row, label in zip(rows, labels):
            filtered_array.append(row)
            feature_mapping.append({
                "index": feature_index,
                "source": "extra",
                "variable": var_name,
                "sub_feature": label,
                "feature_name": f"{var_name}_{label}" if label != "scalar" else var_name,
            })
            feature_index += 1

    filtered_array = np.array(filtered_array)

    metadata = {
        "feature_mapping": feature_mapping,
        "parameters": {
            "read_from_variable": read_from_variable,
            "keypoints_used": list(keypoints),
            "keypoints_available": list(original_keypoints),
            "centered_reference_keypoint": centered_reference_keypoint,
            "orientation_reference_keypoint": orientation_reference_keypoint,
            "excluded_features": excluded_features,
            "extra_features_used": resolved_extra,
            "total_features": len(feature_mapping),
            "data_shape": filtered_array.shape,
        }
    }

    return filtered_array, metadata
