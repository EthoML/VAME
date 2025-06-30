---
sidebar_label: report
title: visualization.report
---

#### logger\_config

#### logger

#### generate\_reports

```python
@save_state(model=GenerateReportsFunctionSchema)
def generate_reports(config: dict,
                     show_figure: bool = True,
                     save_to_file: bool = True,
                     save_logs: bool = True) -> None
```

Generate reports and UMAP for all sessions in the project.

**Parameters**

* **config** (`dict`): Configuration parameters.
* **save_logs** (`bool, optional`): Whether to save logs. Defaults to True.

**Returns**

* `None`

#### report

```python
def report(config: dict,
           segmentation_algorithm: str = "hmm",
           save_to_file: bool = True,
           show_figure: bool = True,
           save_logs: bool = True) -> None
```

Report for a project.

**Parameters**

* **config** (`dict`): Configuration parameters.
* **segmentation_algorithm** (`str, optional`): Segmentation algorithm to use. Defaults to &quot;hmm&quot;.
* **save_to_file** (`bool, optional`): Whether to save the report to file. Defaults to True.
* **show_figure** (`bool, optional`): Whether to show the figure. Defaults to True.
* **save_logs** (`bool, optional`): Whether to save logs. Defaults to True.

**Returns**

* `None`

#### plot\_community\_motifs

```python
def plot_community_motifs(motif_labels,
                          community_labels,
                          community_bag,
                          title: str = "Community and Motif Counts",
                          show_figure: bool = True,
                          save_to_file: bool = False,
                          save_path: str = "") -> None
```

Generates a bar plot to represent community and motif counts with percentages.

