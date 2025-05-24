---
sidebar_label: community
title: visualization.community
---

#### hierarchy\_pos

```python
def hierarchy_pos(G: nx.Graph,
                  root: str | None = None,
                  width: float = 0.5,
                  vert_gap: float = 0.2,
                  vert_loc: float = 0,
                  xcenter: float = 0.5) -> Dict[str, Tuple[float, float]]
```

Positions nodes in a tree-like layout.
Ref: From Joel&#x27;s answer at https://stackoverflow.com/a/29597209/2966723.

**Parameters**

* **G** (`nx.Graph`): The input graph. Must be a tree.
* **root** (`str, optional`): The root node of the tree. If None, the function selects a root node based on graph type.
Defaults to None.
* **width** (`float, optional`): The horizontal space assigned to each level. Defaults to 0.5.
* **vert_gap** (`float, optional`): The vertical gap between levels. Defaults to 0.2.
* **vert_loc** (`float, optional`): The vertical location of the root node. Defaults to 0.
* **xcenter** (`float, optional`): The horizontal location of the root node. Defaults to 0.5.

**Returns**

* `Dict[str, Tuple[float, float]]`: A dictionary mapping node names to their positions (x, y).

#### draw\_tree

```python
def draw_tree(T: nx.Graph,
              fig_width: float = 20.0,
              usage_dict: Dict[str, float] = dict(),
              save_to_file: bool = True,
              show_figure: bool = False,
              results_dir: str | None = None) -> None
```

Draw a tree.

**Parameters**

* **T** (`nx.Graph`): The tree to be drawn.
* **fig_width** (`int, optional`): The width of the figure. Defaults to 20.
* **usage_dict** (`Dict[str, float], optional`): Dictionary mapping node names to their usage values. Defaults to empty dictionary.
* **save_to_file** (`bool, optional`): Flag indicating whether to save the plot. Defaults to True.
* **show_figure** (`bool, optional`): Flag indicating whether to show the plot. Defaults to False.
* **results_dir** (`str, optional`): The directory to save the plot. Defaults to None.

**Returns**

* `None`

#### visualize\_hierarchical\_tree

```python
def visualize_hierarchical_tree(
        config: dict,
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm") -> None
```

Visualizes the hierarchical tree.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **segmentation_algorithm** (`Literal["hmm", "kmeans"]`): Segmentation algorithm. Defaults to &quot;hmm&quot;.

**Returns**

* `None`

