---
sidebar_label: data_manipulation
title: util.data_manipulation
---

#### logger\_config

#### logger

#### consecutive

```python
def consecutive(data: np.ndarray, stepsize: int = 1) -> List[np.ndarray]
```

Find consecutive sequences in the data array.

**Parameters**

* **data** (`np.ndarray`): Input array.
* **stepsize** (`int, optional`): Step size. Defaults to 1.

**Returns**

* `List[np.ndarray]`: List of consecutive sequences.

#### nan\_helper

```python
def nan_helper(y: np.ndarray) -> Tuple
```

Identifies indices of NaN values in an array and provides a function to convert them to non-NaN indices.

**Parameters**

* **y** (`np.ndarray`): Input array containing NaN values.

**Returns**

* `Tuple[np.ndarray, Union[np.ndarray, None]]`: A tuple containing two elements:
- An array of boolean values indicating the positions of NaN values.
- A lambda function to convert NaN indices to non-NaN indices.

#### interpol\_first\_rows\_nans

```python
def interpol_first_rows_nans(arr: np.ndarray) -> np.ndarray
```

Interpolates NaN values in the given array.

**Parameters**

* **arr** (`np.ndarray`): Input array with NaN values.

**Returns**

* `np.ndarray`: Array with interpolated NaN values.

#### interpolate\_nans\_with\_pandas

```python
def interpolate_nans_with_pandas(data: np.ndarray) -> np.ndarray
```

Interpolate NaN values along the time axis of a 3D NumPy array using Pandas.

**Parameters**

* **data** (`numpy.ndarray`): Input 3D array of shape (time, keypoints, space).

**Returns**

* `numpy.ndarray:`: Array with NaN values interpolated.

#### crop\_and\_flip\_legacy

```python
def crop_and_flip_legacy(
        rect: Tuple, src: np.ndarray, points: List[np.ndarray],
        ref_index: Tuple[int, int]) -> Tuple[np.ndarray, List[np.ndarray]]
```

Crop and flip the image based on the given rectangle and points.

**Parameters**

* **rect** (`Tuple`): Rectangle coordinates (center, size, theta).
* **src: np.ndarray**: Source image.
* **points** (`List[np.ndarray]`): List of points.
* **ref_index** (`Tuple[int, int]`): Reference indices for alignment.

**Returns**

* `Tuple[np.ndarray, List[np.ndarray]]`: Cropped and flipped image, and shifted points.

#### background

```python
def background(project_path: str,
               session: str,
               video_path: str,
               num_frames: int = 1000,
               save_background: bool = True) -> np.ndarray
```

Compute background image from fixed camera.

**Parameters**

* **project_path** (`str`): Path to the project directory.
* **session** (`str`): Name of the session.
* **video_path** (`str`): Path to the video file.
* **num_frames** (`int, optional`): Number of frames to use for background computation. Defaults to 1000.

**Returns**

* `np.ndarray`: Background image.

