---
sidebar_label: to_model
title: preprocessing.to_model
---

#### format\_xarray\_for\_rnn

```python
def format_xarray_for_rnn(
        ds: xr.Dataset,
        read_from_variable: str = "position_processed",
        keypoints: list[str] | None = None
) -> Tuple[np.ndarray, Dict[str, Any]]
```

Formats the xarray dataset for use VAME&#x27;s RNN model:
- The x and y coordinates of the centered_reference_keypoint are excluded.
- The x coordinate of the orientation_reference_keypoint is excluded.
- The remaining data is flattened and transposed.

**Parameters**

* **ds** (`xr.Dataset`): The xarray dataset to format.
* **read_from_variable** (`str, default="position_processed"`): The variable to read from the dataset.
* **keypoints** (`list[str] | None, optional`): A list of keypoints to include in the output. If None, all keypoints are
included. If provided, only the specified keypoints will be included in the output.

**Returns**

* `Tuple[np.ndarray, Dict[str, Any]]`: A tuple containing:
- The formatted array in the shape (n_features, n_samples)
- A dictionary with feature provenance and processing information
Where n_features = 2 * n_keypoints * n_spaces - 3.

