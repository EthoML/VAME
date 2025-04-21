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
def umap_vis(embed: np.ndarray, num_points: int) -> plt.Figure
```

Visualize UMAP embedding without labels.

**Parameters**

* **embed** (`np.ndarray`): UMAP embedding.
* **num_points** (`int`): Number of data points to visualize.

**Returns**

* `plt.Figure`: Plot Visualization of UMAP embedding.

#### umap\_label\_vis

```python
def umap_label_vis(embed: np.ndarray, label: np.ndarray,
                   num_points: int) -> plt.Figure
```

Visualize UMAP embedding with motif labels.

**Parameters**

* **embed** (`np.ndarray`): UMAP embedding.
* **label** (`np.ndarray`): Motif labels.
* **num_points** (`int`): Number of data points to visualize.

**Returns**

* `plt.Figure`: Plot figure of UMAP visualization embedding with motif labels.

#### umap\_vis\_comm

```python
def umap_vis_comm(embed: np.ndarray, community_label: np.ndarray,
                  num_points: int) -> plt.Figure
```

Visualize UMAP embedding with community labels.

**Parameters**

* **embed** (`np.ndarray`): UMAP embedding.
* **community_label** (`np.ndarray`): Community labels.
* **num_points** (`int`): Number of data points to visualize.

**Returns**

* `plt.Figure`: Plot figure of UMAP visualization embedding with community labels.

#### visualize\_umap

```python
@save_state(model=VisualizeUmapFunctionSchema)
def visualize_umap(config: dict,
                   segmentation_algorithm: SegmentationAlgorithms,
                   label: Optional[str] = None,
                   save_logs: bool = False) -> None
```

Visualize UMAP embeddings based on configuration settings.
Fills in the values in the &quot;visualization&quot; key of the states.json file.
Saves results files at:

If label is None (UMAP visualization without labels):
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
* **segmentation_algorithm** (`SegmentationAlgorithms`): Which segmentation algorithm to use. Options are &#x27;hmm&#x27; or &#x27;kmeans&#x27;.
* **label** (`str, optional`): Type of labels to visualize. Options are None, &#x27;motif&#x27; or &#x27;community&#x27;. Default is None.
* **save_logs** (`bool, optional`): Save logs to file. Default is False.

**Returns**

* `None`

