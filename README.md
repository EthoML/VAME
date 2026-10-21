![image](https://github.com/EthoML/VAME/assets/844306/0f08424f-06ab-48e4-8094-da0f0c78a08d)

<p align="center">
  <a href="https://codecov.io/gh/EthoML/VAME"><img src="https://codecov.io/gh/EthoML/VAME/graph/badge.svg?token=J1CUXB4N0E"/></a>
  <a href="https://pypi.org/project/vame-py"><img src="https://img.shields.io/pypi/v/vame-py?color=%231BA331&label=PyPI&logo=python&logoColor=%23F7F991%20"/></a>
  <a href="https://ethoml.github.io/VAME/"><img src="https://img.shields.io/badge/docs-ethoml.github.io%2FVAME-blue"/></a>
  <a href="https://github.com/EthoML/VAME/blob/main/LICENSE"><img src="https://img.shields.io/github/license/EthoML/VAME"/></a>
</p>

VAME (Variational Animal Motion Embedding) is an open-source machine learning tool for behavioral action segmentation and analyses.

Read the [documentation](https://ethoml.github.io/VAME/) and the [open-access 2024 Resource article in Cell Reports](https://www.cell.com/cell-reports/fulltext/S2211-1247(24)01221-X).

We are a group of behavioral enthusiasts, comprising the original VAME developers Kevin Luxem and Pavol Bauer; behavioral neuroscientists Stephanie R. Miller, Jorge J. Palop; statisticians Alex Pico, Reuben Thomas; machine learning researchers Pranav Nambiar, Katie Ly, Shreya Bangera, Dakota Mallen, Zhihao Cheng; and CatalystNeuro software engineer team Ben Dichter and Luiz Tauffer. Our aim is to provide scalable, unbiased and sensitive approaches for assessing mouse behavior using computer vision and machine learning approaches.

We are focused on expanding the analytical capabilities of VAME segmentation by providing curated scripts for VAME implementation and tools for data processing, visualization, and statistical analyses.

## VAME in a Nutshell

VAME is a framework to cluster behavioral signals obtained from pose-estimation tools. It is a [PyTorch](https://pytorch.org/)-based deep learning framework which leverages the power of recurrent neural networks (RNN) to model sequential data. In order to learn the underlying complex data distribution, we use the RNN in a variational autoencoder setting to extract the latent state of the animal in every step of the input time series.
The [step-by-step guide](https://ethoml.github.io/VAME/docs/getting_started/step_by_step/) walks through each step of the VAME workflow.

## Main Features

* Reads pose estimation data from [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), [SLEAP](https://github.com/talmolab/sleap), [LightningPose](https://github.com/paninski-lab/lightning-pose), [NWB](https://www.nwb.org/) files (via the [ndx-pose](https://github.com/rly/ndx-pose) extension) and [movement](https://github.com/neuroinformatics-unit/movement) datasets
* Streams NWB pose data directly from [DANDI](https://dandiarchive.org/), see [Read from DANDI](https://ethoml.github.io/VAME/docs/how_to/read_from_dandi/)
* Preprocessing: low-confidence cleaning, egocentric alignment, outlier cleaning, Savitzky-Golay filtering and body-size rescaling
* Learns a low-dimensional latent representation of pose estimation time series with a recurrent variational autoencoder
* Segmentation into behavioral motifs with HMM or k-means
* Hierarchical community analysis of motifs
* Visualizations: UMAP embeddings, motif and community videos, community trees and reports
* Export of results to NWB (via the [ndx-vame](https://github.com/catalystneuro/ndx-vame) extension), see [Export results](https://ethoml.github.io/VAME/docs/how_to/export/)

## Installation

VAME requires Python 3.12 or higher.

```bash
pip install vame-py
```

To read pose data from DANDI, install the `dandi` extra:

```bash
pip install "vame-py[dandi]"
```

See the [installation guide](https://ethoml.github.io/VAME/docs/getting_started/installation/) for installing from source.

## Getting Started

Follow the [step-by-step guide](https://ethoml.github.io/VAME/docs/getting_started/step_by_step/), or run the whole workflow with the [pipeline](https://ethoml.github.io/VAME/docs/getting_started/pipeline/). Both are also available as notebooks in the [examples folder](https://github.com/EthoML/VAME/tree/main/examples).

VAME trains on a GPU when one is available (CUDA or Apple MPS) and falls back to the CPU otherwise.

## VAME App

[VAME App](https://github.com/EthoML/vame-app) is a web application for running the VAME workflow without writing code. It runs locally, calls this library from a Python backend, and is used from the browser. Install it with `pip install vame-app`. See the [VAME App documentation](https://ethoml.github.io/VAME/docs/vame_app/) for details.

## Authors and Code Contributors

VAME was developed by Kevin Luxem and Pavol Bauer (Luxem et al., 2022). The original VAME repository was deprecated, forked, and is now being maintained here at https://github.com/EthoML/VAME.

The development of VAME is heavily inspired by [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/). As such, the VAME project management codebase has been adapted from the DeepLabCut codebase. The DeepLabCut 2.0 toolbox is © A. & M.W. Mathis Labs [deeplabcut.org](https://www.deeplabcut.org), released under LGPL v3.0. The implementation of the VRAE model is partially adapted from the [Timeseries clustering](https://github.com/tejaslodaya/timeseries-clustering-vae) repository developed by [Tejas Lodaya](https://tejaslodaya.com).

## Citing VAME

If you use VAME in your work, please cite:

* Miller, S. R., Luxem, K., Lauderdale, K., et al. (2024). Machine learning reveals prominent spontaneous behavioral changes and treatment efficacy in humanized and transgenic Alzheimer's disease models. *Cell Reports*, 43(11), 114870. https://doi.org/10.1016/j.celrep.2024.114870
* Luxem, K., Mocellin, P., Fuhrmann, F., Kürsch, J., Miller, S. R., Palop, J. J., Remy, S., & Bauer, P. (2022). Identifying behavioral structure from deep variational embeddings of animal motion. *Communications Biology*, 5, 1267. https://doi.org/10.1038/s42003-022-04080-7

## Related Work

* Mocellin et al.: [A septal-ventral tegmental area circuit drives exploratory behavior](https://www.cell.com/neuron/fulltext/S0896-6273(23)00975-3)
* Kingma & Welling: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Support

Report bugs and ask questions in the [issue tracker](https://github.com/EthoML/VAME/issues).

## License

VAME is released under the GPLv3 license. See the [LICENSE file](https://github.com/EthoML/VAME/blob/main/LICENSE) for the full statement.
