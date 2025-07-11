---
sidebar_label: pose_segmentation
title: analysis.pose_segmentation
---

#### logger\_config

#### logger

#### embed\_latent\_vectors

```python
def embed_latent_vectors(
        config: dict,
        sessions: List[str],
        fixed: bool,
        read_from_variable: str = "position_processed",
        overwrite: bool = False,
        tqdm_stream: Union[TqdmToLogger, None] = None) -> List[np.ndarray]
```

Embed latent vectors for the given sessions using the VAME model.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **fixed** (`bool`): Whether the model is fixed.
* **read_from_variable** (`str, optional`): Variable to read from the dataset. Defaults to &quot;position_processed&quot;.
* **overwrite** (`bool, optional`): Whether to overwrite existing latent vector files. Defaults to False.
* **tqdm_stream** (`TqdmToLogger, optional`): TQDM Stream to redirect the tqdm output to logger.

**Returns**

* `List[np.ndarray]`: List of latent vectors for all sessions.

#### embed\_latent\_vectors\_optimized

```python
def embed_latent_vectors_optimized(
        config: dict,
        sessions: List[str],
        fixed: bool,
        read_from_variable: str = "position_processed",
        overwrite: bool = False,
        batch_size: int = 64,
        tqdm_stream: Union[TqdmToLogger, None] = None) -> List[np.ndarray]
```

Optimized version of embed_latent_vectors with batch processing and vectorized operations.

This function provides significant performance improvements over the original implementation:
- Vectorized sliding window creation (no data copying)
- Batch processing of multiple windows simultaneously
- GPU memory optimization with pre-allocated tensors
- Model optimizations for faster inference

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **fixed** (`bool`): Whether the model is fixed.
* **read_from_variable** (`str, optional`): Variable to read from the dataset. Defaults to &quot;position_processed&quot;.
* **overwrite** (`bool, optional`): Whether to overwrite existing latent vector files. Defaults to False.
* **batch_size** (`int, optional`): Number of windows to process simultaneously. Defaults to 64.
Larger values use more GPU memory but may be faster.
* **tqdm_stream** (`TqdmToLogger, optional`): TQDM Stream to redirect the tqdm output to logger.

**Returns**

* `List[np.ndarray]`: List of latent vectors for all sessions.

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

#### save\_session\_data

```python
def save_session_data(project_path: str, session: str, model_name: str,
                      label: np.ndarray, cluster_centers: np.ndarray,
                      motif_usage: np.ndarray, n_clusters: int,
                      segmentation_algorithm: str)
```

Saves pose segmentation data for given session.

**Parameters**

* **project_path: str**: Path to the vame project folder.
* **session: str**: Session name.
* **model_name: str**: Name of model
* **label: np.ndarray**: Array of the session&#x27;s motif labels.
* **cluster_centers: np.ndarray**: Array of the session&#x27;s kmeans cluster centers location in the latent space.
* **motif_usage: np.ndarray**: Array of the session&#x27;s motif usage counts.
* **n_clusters** (`int`): Number of clusters.
* **segmentation_algorithm: str**: Type of segmentation method, either &#x27;kmeans or &#x27;hmm&#x27;.

**Returns**

* `None`

#### same\_segmentation

```python
def same_segmentation(config: dict, sessions: List[str],
                      latent_vectors: List[np.ndarray], n_clusters: int,
                      segmentation_algorithm: str) -> None
```

Apply the same segmentation to all animals.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **latent_vectors** (`List[np.ndarray]`): List of latent vector arrays.
* **n_clusters** (`int`): Number of clusters.
* **segmentation_algorithm** (`str`): Segmentation algorithm.

**Returns**

* `None`

#### individual\_segmentation

```python
def individual_segmentation(config: dict, sessions: List[str],
                            latent_vectors: List[np.ndarray],
                            n_clusters: int) -> Tuple
```

Apply individual segmentation to each session.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **latent_vectors** (`List[np.ndarray]`): List of latent vector arrays.
* **n_clusters** (`int`): Number of clusters.

**Returns**

* `Tuple`: Tuple of labels, cluster centers, and motif usages.

#### segment\_session

```python
@save_state(model=SegmentSessionFunctionSchema)
def segment_session(config: dict,
                    overwrite_segmentation: bool = False,
                    overwrite_embeddings: bool = False,
                    save_logs: bool = True,
                    optimized: bool = True) -> None
```

Perform pose segmentation using the VAME model.
Fills in the values in the &quot;segment_session&quot; key of the states.json file.
Creates files at:
- project_name/
    - results/
        - hmm_trained.pkl
        - session/
            - model_name/
                - latent_vectors.npy
                - hmm-n_clusters/
                    - motif_usage_session.npy
                    - n_cluster_label_session.npy
                - kmeans-n_clusters/
                    - motif_usage_session.npy
                    - n_cluster_label_session.npy
                    - cluster_center_session.npy

latent_vectors.npy contains the projection of the data into the latent space,
for each frame of the video. Dimmentions: (n_frames, n_latent_features)

motif_usage_session.npy contains the number of times each motif was used in the video.
Dimmentions: (n_motifs,)

n_cluster_label_session.npy contains the label of the cluster assigned to each frame.
Dimmentions: (n_frames,)

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **overwrite_segmentation** (`bool, optional`): Whether to overwrite existing segmentation results. Defaults to False.
* **overwrite_embeddings** (`bool, optional`): If True, runs embedding function and re-creates embeddings files, even if they already exist.
Defaults to False.
* **optimized** (`bool, optional`): If True, uses the optimized version of the embedding function.
If False, uses the original version. Defaults to True.
* **save_logs** (`bool, optional`): Whether to save logs. Defaults to True.

**Returns**

* `None`

