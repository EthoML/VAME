---
sidebar_label: visualization
title: preprocessing.visualization
---

#### visualize\_preprocessing\_scatter

```python
def visualize_preprocessing_scatter(
        config: dict,
        session_index: int = 0,
        frames: list = [],
        original_positions_key: str = "position",
        cleaned_positions_key: str = "position_cleaned_lowconf",
        aligned_positions_key: str = "position_egocentric_aligned",
        save_to_file: bool = False,
        show_figure: bool = True)
```

Visualize the preprocessing results by plotting the original, cleaned low-confidence,
and egocentric aligned positions of the keypoints in a scatter plot.

#### visualize\_preprocessing\_timeseries

```python
def visualize_preprocessing_timeseries(
        config: dict,
        session_index: int = 0,
        n_samples: int = 1000,
        original_positions_key: str = "position",
        aligned_positions_key: str = "position_egocentric_aligned",
        processed_positions_key: str = "position_processed",
        save_to_file: bool = False,
        show_figure: bool = True)
```

Visualize the preprocessing results by plotting the original, aligned, and processed positions
of the keypoints in a timeseries plot.

