---
sidebar_label: preprocessing
title: visualization.preprocessing
---

#### logger\_config

#### logger

#### preprocessing\_visualization

```python
@save_state(model=PreprocessingVisualizationFunctionSchema)
def preprocessing_visualization(config: dict,
                                save_to_file: bool = False,
                                show_figure: bool = True) -> None
```

#### visualize\_preprocessing\_scatter

```python
def visualize_preprocessing_scatter(
        config: dict,
        session_index: int = 0,
        frames: list = [],
        original_positions_key: str | None = "position",
        cleaned_positions_key: str | None = "position_cleaned_lowconf",
        aligned_positions_key: str | None = "position_egocentric_aligned",
        filtered_positions_key: str | None = "position_processed",
        scaled_positions_key: str | None = "position_scaled",
        save_to_file: bool = False,
        show_figure: bool = True)
```

Visualize the preprocessing results by plotting the positions of the keypoints in a scatter plot.
Each position key parameter can be a string (to include that column) or None (to skip that column).

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **session_index** (`int, optional`): Index of the session to visualize.
* **frames** (`list, optional`): List of frames to visualize.
* **original_positions_key** (`str, optional`): Key for the original positions.
* **cleaned_positions_key** (`str, optional`): Key for the low confidence cleaned positions.
* **aligned_positions_key** (`str, optional`): Key for the egocentric aligned positions.
* **filtered_positions_key** (`str, optional`): Key for the filtered positions.
* **scaled_positions_key** (`str, optional`): Key for the scaled positions.
* **save_to_file** (`bool, optional`): Whether to save the figure to a file.
* **show_figure** (`bool, optional`): Whether to show the figure.

**Returns**

* `None`

#### visualize\_preprocessing\_timeseries

```python
def visualize_preprocessing_timeseries(
        config: dict,
        session_index: int = 0,
        n_samples: int = 1000,
        sample_offset: int = 0,
        original_positions_key: str | None = "position",
        aligned_positions_key: str | None = "position_egocentric_aligned",
        filtered_positions_key: str | None = "position_processed",
        scaled_positions_key: str | None = "position_scaled",
        keypoints: list | None = None,
        save_to_file: bool = False,
        show_figure: bool = True)
```

Visualize the preprocessing results by plotting position data in a timeseries plot.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **session_index** (`int, optional`): Index of the session to visualize.
* **n_samples** (`int, optional`): Number of samples to plot.
* **sample_offset** (`int, optional`): Starting index for the time series data. Default is 0 (start from beginning).
* **original_positions_key** (`str | None, optional`): Key for the original positions. If None, this position type will be skipped.
* **aligned_positions_key** (`str | None, optional`): Key for the aligned positions. If None, this position type will be skipped.
* **filtered_positions_key** (`str | None, optional`): Key for the filtered positions. If None, this position type will be skipped.
* **scaled_positions_key** (`str | None, optional`): Key for the scaled positions. If None, this position type will be skipped.
* **keypoints** (`list | None, optional`): List of keypoint names to include in the visualization. If None or empty list,
all keypoints will be included.
* **save_to_file** (`bool, optional`): Whether to save the figure to a file.
* **show_figure** (`bool, optional`): Whether to show the figure.

**Returns**

* `None`

#### visualize\_preprocessing\_cloud

```python
def visualize_preprocessing_cloud(
        config: dict,
        session_index: int = 0,
        n_samples: int = 1000,
        aligned_positions_key: str | None = "position_egocentric_aligned",
        filtered_positions_key: str | None = "position_processed",
        scaled_positions_key: str | None = "position_scaled",
        keypoints: list | None = None,
        alpha: float = 0.3,
        save_to_file: bool = False,
        show_figure: bool = True)
```

Visualize the preprocessing results by plotting a cloud of keypoint positions across multiple frames.
Only includes aligned, filtered, and scaled positions as these are in comparable coordinate systems.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **session_index** (`int, optional`): Index of the session to visualize.
* **n_samples** (`int, optional`): Number of frames to include in the visualization. Frames are randomly sampled.
* **aligned_positions_key** (`str | None, optional`): Key for the egocentric aligned positions. If None, this position type will be skipped.
* **filtered_positions_key** (`str | None, optional`): Key for the filtered positions. If None, this position type will be skipped.
* **scaled_positions_key** (`str | None, optional`): Key for the scaled positions. If None, this position type will be skipped.
* **keypoints** (`list | None, optional`): List of keypoint names to include. If None, all keypoints will be included.
* **alpha** (`float, optional`): Transparency level for the dots (0.0 to 1.0).
* **save_to_file** (`bool, optional`): Whether to save the figure to a file.
* **show_figure** (`bool, optional`): Whether to show the figure.

**Returns**

* `None`

