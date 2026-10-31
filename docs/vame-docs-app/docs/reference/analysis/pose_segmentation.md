---
sidebar_label: pose_segmentation
title: analysis.pose_segmentation
---

#### logger\_config

#### logger

#### sliding\_windows

```python
def sliding_windows(data: np.ndarray, temp_win: int) -> np.ndarray
```

View of all windows of temp_win frames, shape (n_windows, temp_win, n_features),
from data of shape (n_features, n_frames). Window i covers frames i to i + temp_win - 1.

#### check\_input\_distribution

```python
def check_input_distribution(data: np.ndarray, session: str) -> None
```

Warn when a session&#x27;s normalized input is far from the training distribution (mean 0, std 1).

#### embed\_latent\_vectors

```python
def embed_latent_vectors(
        config: dict,
        sessions: List[str],
        fixed: bool,
        read_from_variable: str | None = None,
        overwrite: bool = False,
        batch_size: int = 64,
        tqdm_stream: Union[TqdmToLogger, None] = None) -> List[np.ndarray]
```

Embed each session&#x27;s sliding windows into the latent space, encoding windows in batches.
Saves the latent vectors to results/&lt;session&gt;/&lt;model_name&gt;/latent_vectors.npy.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **sessions** (`List[str]`): List of session names.
* **fixed** (`bool`): Whether the model is fixed.
* **read_from_variable** (`str, optional`): Variable to read from the dataset. Defaults to the variable recorded in
the training metadata, so inference uses the same input as training.
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
* **segmentation_algorithm: str**: Type of segmentation method: &#x27;kmeans&#x27;, &#x27;hmm&#x27;, or &#x27;hmm_warmstart&#x27;.

**Returns**

* `None`

#### \_session\_estep

```python
def _session_estep(model: "hmm.GaussianHMM",
                   session_data: np.ndarray) -> Tuple[dict, float]
```

The expensive, session-independent half of one Baum-Welch iteration:
forward-backward + sufficient-statistics accumulation for a single
session, against the model&#x27;s current parameters. Designed to be run in
parallel across sessions via joblib - safe to do so because sufficient
statistics are purely additive across independent sequences (this is the
same assumption hmmlearn&#x27;s own sequential per-sequence loop relies on).

#### \_merge\_stats

```python
def _merge_stats(stats_list: List[dict]) -> dict
```

Sum per-session sufficient statistics into one combined stats dict.
Values are either plain int (nobs) or np.ndarray; `+` handles both and
always returns a new object, so no explicit .copy() is needed.

#### parallel\_gaussian\_hmm\_fit

```python
def parallel_gaussian_hmm_fit(model: "hmm.GaussianHMM",
                              session_arrays: List[np.ndarray],
                              n_jobs: int = 1) -> "hmm.GaussianHMM"
```

Equivalent to model.fit(np.concatenate(session_arrays), lengths=[...]),
with the E-step (forward-backward per session) parallelized across
processes instead of hmmlearn&#x27;s own single-threaded per-sequence loop.
Mirrors hmmlearn.base.BaseHMM.fit() exactly (same _init/_check/monitor_
calls) - only how the E-step&#x27;s stats/logprob get computed differs.

**Parameters**

* **model** (`hmm.GaussianHMM`): A configured (not yet fit) GaussianHMM instance.
* **session_arrays** (`List[np.ndarray]`): Per-session latent vector arrays.
* **n_jobs** (`int, optional`): Passed to joblib.Parallel. 1 = sequential (no process overhead),
-1 = use all available cores. Defaults to 1.

**Returns**

* `hmm.GaussianHMM`: The same model instance, fit in place.

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

#### segment\_session

```python
@save_state(model=SegmentSessionFunctionSchema)
def segment_session(config: dict,
                    overwrite_segmentation: bool = False,
                    overwrite_embeddings: bool = False,
                    save_logs: bool = True) -> None
```

Perform pose segmentation using the VAME model.
Fills in the values in the &quot;segment_session&quot; key of the states.json file.
Creates files at:
- project_name/
    - results/
        - hmm_trained.pkl
        - hmm_warmstart_trained.pkl
        - session/
            - model_name/
                - latent_vectors.npy
                - hmm-n_clusters/
                    - motif_usage_session.npy
                    - n_cluster_label_session.npy
                - hmm_warmstart-n_clusters/
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
* **save_logs** (`bool, optional`): Whether to save logs. Defaults to True.

**Returns**

* `None`

