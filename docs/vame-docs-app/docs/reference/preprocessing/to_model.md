---
sidebar_label: to_model
title: preprocessing.to_model
---

#### format\_xarray\_for\_rnn

```python
def format_xarray_for_rnn(ds: xr.Dataset,
                          read_from_variable: str = "position_processed")
```

Formats the xarray dataset for use VAME&#x27;s RNN model:
- The x and y coordinates of the centered_reference_keypoint are excluded.
- The x coordinate of the orientation_reference_keypoint is excluded.
- The remaining data is flattened and transposed.

**Parameters**

* **ds** (`xr.Dataset`): The xarray dataset to format.
* **read_from_variable** (`str, default="position_processed"`): The variable to read from the dataset.

**Returns**

* `np.ndarray`: The formatted array in the shape (n_features, n_samples).
Where n_features = 2 * n_keypoints * n_spaces - 3.

