---
sidebar_label: umap
title: visualization.umap
---

#### logger\_config

#### logger

#### umap\_embedding

```python
def umap_embedding(config: dict, num_points: int = 30_000) -> xr.Dataset
```

Perform UMAP embedding for a sample of the entire project.

**Parameters**

* **config** (`dict`): Configuration parameters.
* **num_points** (`int`): Number of points to use for UMAP embedding. Default is 30,000.

**Returns**

* `xr.Dataset`: UMAP embedding and accompanying data for the sampled indices.

#### umap\_vis\_matplotlib

```python
def umap_vis_matplotlib(embed: np.ndarray,
                        num_points: int = 30_000,
                        labels: Optional[np.ndarray] = None,
                        title: Optional[str] = None,
                        show_legend: bool = True,
                        label_type: str = "none") -> Figure
```

Visualize UMAP embedding.

**Parameters**

* **embed** (`np.ndarray`): UMAP embedding.
* **num_points** (`int`): Number of data points to visualize. Default is 30,000.
* **labels** (`np.ndarray, optional`): Motif or community labels. Default is None.
* **title** (`str, optional`): Title for the plot. Default is None.
* **show_legend** (`bool, optional`): Whether to show legend for labeled plots. Default is True.
* **label_type** (`str, optional`): Type of labels (&#x27;none&#x27;, &#x27;motif&#x27;, &#x27;community&#x27;). Default is &#x27;none&#x27;.

**Returns**

* `Figure`: Matplotlib figure object.

#### umap\_vis\_plotly

```python
def umap_vis_plotly(embed: np.ndarray,
                    labels_motif: Optional[np.ndarray] = None,
                    labels_community: Optional[np.ndarray] = None,
                    session_names: Optional[list] = None,
                    num_points: int = 30_000,
                    title: str = "UMAP",
                    marker_size: float = 3.5,
                    marker_opacity: float = 0.7) -> go.Figure
```

Create an interactive Plotly UMAP scatter with dropdown to select labels:
  - None: grey points
  - Motif: colored by motif labels
  - Community: colored by community labels
  - Session: colored by session labels

**Parameters**

* **embed** (`np.ndarray`): 2D UMAP embedding array of shape (N,2).
* **labels_motif** (`np.ndarray or None`): 1D motif labels of length N.
* **labels_community** (`np.ndarray or None`): 1D community labels of length N.
* **session_names** (`list or None`): List of session names for each point.
* **num_points** (`int`): Maximum number of points to show.
* **title** (`str`): Title for the figure. Defaults to &quot;UMAP&quot;.
* **marker_size** (`float`): Size of the markers in the plot.
* **marker_opacity** (`float`): Opacity of the markers in the plot.

**Returns**

* `plotly.graph_objs.Figure`: The interactive Plotly figure.

#### visualize\_umap

```python
def visualize_umap(config: dict,
                   num_points: int = 30_000,
                   save_to_file: bool = True,
                   show_figure: Literal["none", "matplotlib", "plotly",
                                        "all"] = "none",
                   save_logs: bool = True) -> None
```

Visualize UMAP embeddings based on configuration settings.
Fills in the values in the &quot;visualization_umap&quot; key of the states.json file.
Saves results files at:
- project_name/
    - results/
        - umap_embedding.nc
    - reports/
        - umap/
            - umap_{model}_{seg}-{n_clusters}.png              (UMAP visualization without labels)
            - umap_{model}_{seg}-{n_clusters}_motif.png        (UMAP visualization with motif labels)
            - umap_{model}_{seg}-{n_clusters}_community.png    (UMAP visualization with community labels)
            - umap_{model}_{seg}-{n_clusters}_interactive.html (Interactive UMAP visualization)

**Parameters**

* **config** (`dict`): Configuration parameters.
* **num_points** (`int, optional`): Number of data points to visualize. Default is 30,000.
* **save_to_file** (`bool, optional`): Save the figure to file. Default is True.
* **show_figure** (`Literal["none", "matplotlib", "plotly", "all"], optional`): Show the figure. Default is &quot;none&quot;.
- &quot;none&quot;: do not show
- &quot;matplotlib&quot;: show with matplotlib
- &quot;plotly&quot;: show with plotly
- &quot;all&quot;: show with both matplotlib and plotly
* **save_logs** (`bool, optional`): Save logs. Default is True.

**Returns**

* `None`

