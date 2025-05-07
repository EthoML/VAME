---
sidebar_label: community_analysis
title: analysis.community_analysis
---

#### logger\_config

#### logger

#### get\_adjacency\_matrix

```python
def get_adjacency_matrix(
        labels: np.ndarray,
        n_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Calculate the adjacency matrix, transition matrix, and temporal matrix.

**Parameters**

* **labels** (`np.ndarray`): Array of cluster labels.
* **n_clusters** (`int`): Number of clusters.

**Returns**

* `Tuple[np.ndarray, np.ndarray, np.ndarray]`: Tuple containing: adjacency matrix, transition matrix, and temporal matrix.

#### get\_transition\_matrix

```python
def get_transition_matrix(adjacency_matrix: np.ndarray,
                          threshold: float = 0.0) -> np.ndarray
```

Compute the transition matrix from the adjacency matrix.

**Parameters**

* **adjacency_matrix** (`np.ndarray`): Adjacency matrix.
* **threshold** (`float, optional`): Threshold for considering transitions. Defaults to 0.0.

**Returns**

* `np.ndarray`: Transition matrix.

#### get\_motif\_labels

```python
def get_motif_labels(
        config: dict, sessions: List[str], model_name: str, n_clusters: int,
        segmentation_algorithm: str) -> Tuple[np.ndarray, np.ndarray]
```

Get motif labels and motif counts for the entire cohort.

**Parameters**

* **config** (`dict`): Configuration parameters.
* **sessions** (`List[str]`): List of session names.
* **model_name** (`str`): Model name.
* **n_clusters** (`int`): Number of clusters.
* **segmentation_algorithm** (`str`): Which segmentation algorithm to use. Options are &#x27;hmm&#x27; or &#x27;kmeans&#x27;.

**Returns**

* `Tuple [np.ndarray, np.ndarray]`: Tuple with:
    - Array of motif labels (integers) of the entire cohort
    - Array of motif counts of the entire cohort

#### compute\_transition\_matrices

```python
def compute_transition_matrices(files: List[str], labels: List[np.ndarray],
                                n_clusters: int) -> List[np.ndarray]
```

Compute transition matrices for given files and labels.

**Parameters**

* **files** (`List[str]`): List of file paths.
* **labels** (`List[np.ndarray]`): List of label arrays.
* **n_clusters** (`int`): Number of clusters.

**Returns**

* `List[np.ndarray]:`: List of transition matrices.

#### create\_cohort\_community\_bag

```python
def create_cohort_community_bag(
        config: dict, motif_labels: List[np.ndarray],
        trans_mat_full: np.ndarray, cut_tree: int | None, n_clusters: int,
        segmentation_algorithm: Literal["hmm", "kmeans"]) -> list
```

Create cohort community bag for given motif labels, transition matrix,
cut tree, and number of clusters. (markov chain to tree -&gt; community detection)

**Parameters**

* **config** (`dict`): Configuration parameters.
* **motif_labels** (`List[np.ndarray]`): List of motif label arrays.
* **trans_mat_full** (`np.ndarray`): Full transition matrix.
* **cut_tree** (`int | None`): Cut line for tree.
* **n_clusters** (`int`): Number of clusters.
* **segmentation_algorithm** (`str`): Which segmentation algorithm to use. Options are &#x27;hmm&#x27; or &#x27;kmeans&#x27;.

**Returns**

* `List`: List of community bags.

#### get\_cohort\_community\_labels

```python
def get_cohort_community_labels(
        motif_labels: List[np.ndarray],
        cohort_community_bag: list) -> List[np.ndarray]
```

Transform kmeans/hmm parameterized latent vector motifs into communities.
Get cohort community labels for given labels, and community bags.

**Parameters**

* **labels** (`List[np.ndarray]`): List of label arrays.
* **cohort_community_bag** (`np.ndarray`): List of community bags. Dimensions: (n_communities, n_clusters_in_community)

**Returns**

* `List[np.ndarray]`: List of cohort community labels for each file.

#### save\_cohort\_community\_labels\_per\_session

```python
def save_cohort_community_labels_per_session(
        config: dict, sessions: List[str], model_name: str, n_clusters: int,
        segmentation_algorithm: str, cohort_community_bag: list) -> None
```

#### community

```python
@save_state(model=CommunityFunctionSchema)
def community(config: dict,
              cut_tree: int | None = None,
              save_logs: bool = True) -> None
```

Perform community analysis.
Fills in the values in the &quot;community&quot; key of the states.json file.
Saves results files at:
- project_name/
    - results/
        - community_cohort/
            - segmentation_algorithm-n_clusters/
                - cohort_community_bag.npy
                - cohort_community_label.npy
                - cohort_segmentation_algorithm_label.npy
                - cohort_transition_matrix.npy
                - hierarchy.pkl
        - session_name/
            - model_name/
                - segmentation_algorithm-n_clusters/
                    - community/
                        - cohort_community_label_session_name.npy

**Parameters**

* **config** (`dict`): Configuration parameters.
* **cut_tree** (`int, optional`): Cut line for tree. Defaults to None.
* **save_logs** (`bool, optional`): Whether to save logs. Defaults to True.

**Returns**

* `None`

