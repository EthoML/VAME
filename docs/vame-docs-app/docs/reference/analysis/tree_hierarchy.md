---
sidebar_label: tree_hierarchy
title: analysis.tree_hierarchy
---

#### merge\_func

```python
def merge_func(transition_matrix: np.ndarray, n_clusters: int,
               motif_norm: np.ndarray,
               merge_sel: int) -> Tuple[np.ndarray, np.ndarray]
```

Merge nodes in a graph based on a selection criterion.

**Parameters**

* **transition_matrix** (`np.ndarray`): The transition matrix of the graph.
* **n_clusters** (`int`): The number of clusters.
* **motif_norm** (`np.ndarray`): The normalized motif matrix.
* **merge_sel** (`int`): The merge selection criterion.
- 0: Merge nodes with highest transition probability.
- 1: Merge nodes with lowest cost.

**Returns**

* `Tuple[np.ndarray, np.ndarray]`: A tuple containing the merged nodes.

#### graph\_to\_tree

```python
def graph_to_tree(motif_usage: np.ndarray,
                  transition_matrix: np.ndarray,
                  n_clusters: int,
                  merge_sel: int = 1) -> nx.Graph
```

Convert a graph to a tree.

**Parameters**

* **motif_usage** (`np.ndarray`): The motif usage matrix.
* **transition_matrix** (`np.ndarray`): The transition matrix of the graph.
* **n_clusters** (`int`): The number of clusters.
* **merge_sel** (`int, optional`): The merge selection criterion. Defaults to 1.
- 0: Merge nodes with highest transition probability.
- 1: Merge nodes with lowest cost.

**Returns**

* `nx.Graph`: The tree.

#### \_traverse\_tree\_cutline

```python
def _traverse_tree_cutline(
        T: nx.Graph,
        node: List[str],
        traverse_list: List[str],
        cutline: int,
        level: int,
        community_bag: List[List[str]],
        community_list: List[str] | None = None) -> List[List[str]]
```

DEPRECATED in favor of bag_nodes_by_cutline.
Helper function for tree traversal with a cutline.

**Parameters**

* **T** (`nx.Graph`): The tree to be traversed.
* **node** (`List[str]`): Current node being traversed.
* **traverse_list** (`List[str]`): List of traversed nodes.
* **cutline** (`int`): The cutline level.
* **level** (`int`): The current level in the tree.
* **community_bag** (`List[List[str]]`): List of community bags.
* **community_list** (`List[str], optional`): List of nodes in the current community bag.

**Returns**

* `List[List[str]]`: List of lists community bags.

#### traverse\_tree\_cutline

```python
def traverse_tree_cutline(T: nx.Graph,
                          root_node: str | None = None,
                          cutline: int = 2) -> List[List[str]]
```

DEPRECATED in favor of bag_nodes_by_cutline.
Traverse a tree with a cutline and return the community bags.

**Parameters**

* **T** (`nx.Graph`): The tree to be traversed.
* **root_node** (`str, optional`): The root node of the tree. If None, traversal starts from the root.
* **cutline** (`int, optional`): The cutline level.

**Returns**

* `List[List[str]]`: List of community bags.

#### bag\_nodes\_by\_cutline

```python
def bag_nodes_by_cutline(tree: nx.Graph, cutline: int = 2, root: str = "Root")
```

Bag nodes of a tree by a cutline.

**Parameters**

* **tree** (`nx.Graph`): The tree to be bagged.
* **cutline** (`int, optional`): The cutline level. Defaults to 2.
* **root** (`str, optional`): The root node of the tree. Defaults to &#x27;Root&#x27;.

**Returns**

* `List[List[str]]`: List of bags of nodes.

