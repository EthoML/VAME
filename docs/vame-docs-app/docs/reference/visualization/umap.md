---
sidebar_label: umap
title: visualization.umap
---

#### logger\_config

#### logger

#### umap\_embedding

```python
def umap_embedding(
        config: dict, session: str, model_name: str, n_clusters: int,
        segmentation_algorithm: SegmentationAlgorithms) -> np.ndarray
```

Perform UMAP embedding for given file and parameters.

**Parameters**

* **config** (`dict`): Configuration parameters.
* **session** (`str`): Session name.
* **model_name** (`str`): Model name.
* **n_clusters** (`int`): Number of clusters.
* **segmentation_algorithm** (`str`): Segmentation algorithm.

**Returns**

* `np.ndarray`: UMAP embedding.

#### umap\_vis

```python
def umap_vis(embed: np.ndarray,
             num_points: int,
             labels: Optional[np.ndarray] = None,
             save_to_file: bool = False,
             show_figure: bool = True) -> Figure
```

Visualize UMAP embedding.

**Parameters**

* **embed** (`np.ndarray`): UMAP embedding.
* **num_points** (`int`): Number of data points to visualize.
* **labels** (`np.ndarray, optional`): Motif or community labels. Default is None.

**Returns**

* `Figure`: Matplotlib figure object.

#### visualize\_umap

```python
def visualize_umap(config: dict,
                   save_to_file: bool = True,
                   show_figure: bool = True,
                   save_logs: bool = True) -> None
```

Visualize UMAP embeddings based on configuration settings.
Fills in the values in the &quot;visualization_umap&quot; key of the states.json file.
Saves results files at:
- project_name/
    - results/
        - file_name/
            - model_name/
                - segmentation_algorithm-n_clusters/
                    - community/
                        - umap_embedding_file_name.npy
                        - umap_vis_label_none_file_name.png  (UMAP visualization without labels)
                        - umap_vis_motif_file_name.png  (UMAP visualization with motif labels)
                        - umap_vis_community_file_name.png  (UMAP visualization with community labels)

**Parameters**

* **config** (`dict`): Configuration parameters.
* **save_to_file** (`bool, optional`): Save the figure to file. Default is True.
* **show_figure** (`bool, optional`): Show the figure. Default is True.
* **save_logs** (`bool, optional`): Save logs. Default is True.

**Returns**

* `None`

