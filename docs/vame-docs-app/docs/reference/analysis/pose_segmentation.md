---
sidebar_label: pose_segmentation
title: analysis.pose_segmentation
---

#### logger\_config

#### logger

#### embedd\_latent\_vectors

```python
def embedd_latent_vectors(
        cfg: dict,
        sessions: List[str],
        model: RNN_VAE,
        fixed: bool,
        read_from_variable: str = "position_processed",
        tqdm_stream: Union[TqdmToLogger, None] = None) -> List[np.ndarray]
```

Embed latent vectors for the given files using the VAME model.

**Parameters**

* **cfg** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **model** (`RNN_VAE`): VAME model.
* **fixed** (`bool`): Whether the model is fixed.
* **tqdm_stream** (`TqdmToLogger, optional`): TQDM Stream to redirect the tqdm output to logger.

**Returns**

* `List[np.ndarray]`: List of latent vectors for each file.

#### get\_motif\_usage

```python
def get_motif_usage(session_labels: np.ndarray, n_clusters: int) -> np.ndarray
```

Count motif usage from session label array.

**Parameters**

* **session_labels** (`np.ndarray`): Array of session labels.
* **n_clusters** (`int`): Number of clusters.

**Returns**

* `np.ndarray`: Array of motif usage counts.

#### same\_segmentation

```python
def same_segmentation(
    cfg: dict, sessions: List[str], latent_vectors: List[np.ndarray],
    n_clusters: int, segmentation_algorithm: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
```

Apply the same segmentation to all animals.

**Parameters**

* **cfg** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **latent_vectors** (`List[np.ndarray]`): List of latent vector arrays.
* **n_clusters** (`int`): Number of clusters.
* **segmentation_algorithm** (`str`): Segmentation algorithm.

**Returns**

* `Tuple`: Tuple of labels, cluster centers, and motif usages.

#### individual\_segmentation

```python
def individual_segmentation(cfg: dict, sessions: List[str],
                            latent_vectors: List[np.ndarray],
                            n_clusters: int) -> Tuple
```

Apply individual segmentation to each session.

**Parameters**

* **cfg** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **latent_vectors** (`List[np.ndarray]`): List of latent vector arrays.
* **n_clusters** (`int`): Number of clusters.

**Returns**

* `Tuple`: Tuple of labels, cluster centers, and motif usages.

#### segment\_session

```python
@save_state(model=SegmentSessionFunctionSchema)
def segment_session(config: dict, save_logs: bool = False) -> None
```

Perform pose segmentation using the VAME model.
Fills in the values in the &quot;segment_session&quot; key of the states.json file.
Creates files at:
- project_name/
    - results/
        - hmm_trained.pkl
        - session/
            - model_name/
                - hmm-n_clusters/
                    - latent_vector_session.npy
                    - motif_usage_session.npy
                    - n_cluster_label_session.npy
                - kmeans-n_clusters/
                    - latent_vector_session.npy
                    - motif_usage_session.npy
                    - n_cluster_label_session.npy
                    - cluster_center_session.npy

latent_vector_session.npy contains the projection of the data into the latent space,
for each frame of the video. Dimmentions: (n_frames, n_latent_features)

motif_usage_session.npy contains the number of times each motif was used in the video.
Dimmentions: (n_motifs,)

n_cluster_label_session.npy contains the label of the cluster assigned to each frame.
Dimmentions: (n_frames,)

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **save_logs** (`bool, optional`): Whether to save logs, by default False.

**Returns**

* `None`

