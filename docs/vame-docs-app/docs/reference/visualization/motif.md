---
sidebar_label: motif
title: visualization.motif
---

#### visualize\_motif\_thresholding

```python
def visualize_motif_thresholding(
    config: dict,
    segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
    n_clusters: int | None = None,
    threshold: float = 1.0,
    show_figure: bool = True,
    save_to_file: bool = False,
    fig_size: tuple = (10, 6)) -> None
```

Visualizes the motif usage with thresholding.

**Parameters**

* **segmentation_algorithm** (`Literal["hmm", "kmeans"], optional`): Segmentation algorithm, by default &quot;hmm&quot;.
* **n_clusters** (`Optional[int], optional`): Number of clusters, by default None. If None, it uses the value from config[&quot;n_clusters&quot;].
* **threshold** (`float, optional`): Threshold, by default 1.0.
* **show_figure** (`bool, optional`): Whether to show the figure, by default True.
* **save_to_file** (`bool, optional`): Whether to save the figure to file, by default False.
* **fig_size** (`tuple, optional`): Figure size, by default (10, 6).

**Returns**

* `None`

