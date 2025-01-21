---
sidebar_label: generative_functions
title: analysis.generative_functions
---

#### logger\_config

#### logger

#### random\_generative\_samples\_motif

```python
def random_generative_samples_motif(cfg: dict, model: torch.nn.Module,
                                    latent_vector: np.ndarray,
                                    labels: np.ndarray,
                                    n_clusters: int) -> plt.Figure
```

Generate random samples for motifs.

**Parameters**

* **cfg** (`dict`): Configuration dictionary.
* **model** (`torch.nn.Module`): PyTorch model.
* **latent_vector** (`np.ndarray`): Latent vectors.
* **labels** (`np.ndarray`): Labels.
* **n_clusters** (`int`): Number of clusters.

**Returns**

* `plt.Figure`: Figure of generated samples.

#### random\_generative\_samples

```python
def random_generative_samples(cfg: dict, model: torch.nn.Module,
                              latent_vector: np.ndarray) -> plt.Figure
```

Generate random generative samples.

**Parameters**

* **cfg** (`dict`): Configuration dictionary.
* **model** (`torch.nn.Module`): PyTorch model.
* **latent_vector** (`np.ndarray`): Latent vectors.

**Returns**

* `plt.Figure`: Figure of generated samples.

#### random\_reconstruction\_samples

```python
def random_reconstruction_samples(cfg: dict, model: torch.nn.Module,
                                  latent_vector: np.ndarray) -> plt.Figure
```

Generate random reconstruction samples.

**Parameters**

* **cfg** (`dict`): Configuration dictionary.
* **model** (`torch.nn.Module`): PyTorch model to use.
* **latent_vector** (`np.ndarray`): Latent vectors.

**Returns**

* `plt.Figure`: Figure of reconstructed samples.

#### visualize\_cluster\_center

```python
def visualize_cluster_center(cfg: dict, model: torch.nn.Module,
                             cluster_center: np.ndarray) -> plt.Figure
```

Visualize cluster centers.

**Parameters**

* **cfg** (`dict`): Configuration dictionary.
* **model** (`torch.nn.Module`): PyTorch model.
* **cluster_center** (`np.ndarray`): Cluster centers.

**Returns**

* `plt.Figure`: Figure of cluster centers.

#### generative\_model

```python
@save_state(model=GenerativeModelFunctionSchema)
def generative_model(config: dict,
                     segmentation_algorithm: SegmentationAlgorithms,
                     mode: str = "sampling",
                     save_logs: bool = False) -> plt.Figure
```

Generative model.

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **mode** (`str, optional`): Mode for generating samples. Defaults to &quot;sampling&quot;.

**Returns**

* `plt.Figure`: Plots of generated samples for each segmentation algorithm.

