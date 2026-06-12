---
sidebar_label: seed
title: util.seed
---

#### seed\_everything

```python
def seed_everything(seed: int) -> None
```

Seed every RNG VAME draws from, so a fixed ``project_random_state`` gives
stable scientific output across runs on the same machine/device.

Seeds Python ``random``, NumPy&#x27;s global RNG and Torch (CPU + CUDA). Call once
at the start of each pipeline op; downstream global-RNG draws (dataset crops,
KMeans/HMM/UMAP, generative and visualization sampling) then become
deterministic. This reseeds the global RNGs as a side effect. It does not
enable Torch deterministic algorithms (kept off to avoid the training
slowdown), so cross-device bit-exactness is not guaranteed.

