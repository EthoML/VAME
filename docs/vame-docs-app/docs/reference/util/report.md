---
sidebar_label: report
title: util.report
---

#### logger\_config

#### logger

#### report

```python
def report(config: dict, segmentation_algorithm: str = "hmm") -> None
```

Report for a project.

#### plot\_community\_motifs

```python
def plot_community_motifs(motif_labels,
                          community_labels,
                          community_bag,
                          title: str = "Community and Motif Counts",
                          save_to_file: bool = False,
                          save_path: str = "")
```

Generates a bar plot to represent community and motif counts with percentages.

