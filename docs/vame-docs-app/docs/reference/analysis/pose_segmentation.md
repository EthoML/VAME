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

Embed latent vectors for the given files using the VAME model.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **fixed** (`bool`): Whether the model is fixed.
* **read_from_variable** (`str, optional`): Variable to read from the dataset. Defaults to &quot;position_processed&quot;.
* **overwrite** (`bool, optional`): Whether to overwrite existing latent vector files. Defaults to False.
* **tqdm_stream** (`TqdmToLogger, optional`): TQDM Stream to redirect the tqdm output to logger.

**Returns**

* `List[np.ndarray]`: List of latent vectors for each file.

#### get\_latent\_vectors

```python
def get_latent_vectors(project_path: str, sessions: list, model_name: str, seg,
                       n_clusters: int) -> List
```

Gets all the latent vectors from each session into one list

**Parameters**

* **project_path: str**: Path to vame project folder
* **session: list**: List of sessions
* **model_name: str**: Name of model
* **seg: str**: Type of segmentation algorithm
* **n_clusters** (`int`): Number of clusters.

**Returns**

* `List`: List of session latent vectors

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
def save_session_data(project_path: str, session: int, model_name: str,
                      label: np.ndarray, cluster_center: np.ndarray,
                      latent_vector: np.ndarray, motif_usage: np.ndarray,
                      n_clusters: int, segmentation_algorithm: str)
```

Saves pose segmentation data for given session.

**Parameters**

* **project_path: str**: Path to the vame project folder.
* **session: int**: Session of interest to segment.
* **model_name: str**: Name of model
* **label: np.ndarray**: Array of the session&#x27;s motif labels.
* **cluster_center: np.ndarray**: Array of the session&#x27;s kmeans cluster centers location in the latent space.
* **latent_vector: np.ndarray,**: Array of the session&#x27;s latent vectors.
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
                    overwrite: bool = False,
                    run_embedding: bool = False,
                    save_logs: bool = True) -> None
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
* **overwrite** (`bool, optional`): Whether to overwrite existing segmentation results. Defaults to False.
* **run_embedding** (`bool, optional`): If True, runs embedding function and re-creates embeddings files, even if they already exist.
Defaults to False.
* **save_logs** (`bool, optional`): Whether to save logs. Defaults to True.

**Returns**

* `None`

