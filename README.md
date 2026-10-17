![image](https://github.com/EthoML/VAME/assets/844306/0f08424f-06ab-48e4-8094-da0f0c78a08d)

<p align="center">
<a href="https://codecov.io/gh/EthoML/VAME" >
 <img src="https://codecov.io/gh/EthoML/VAME/graph/badge.svg?token=J1CUXB4N0E"/>
 </a>
   <a href="https://pypi.org/project/vame-py">
    <img src="https://img.shields.io/pypi/v/vame-py?color=%231BA331&label=PyPI&logo=python&logoColor=%23F7F991%20">
  </a>
</p>

🌟 Welcome to EthoML/VAME (Variational Animal Motion Embedding), an open-source machine learning tool for behavioral action segmentation and analyses.

VAME [documentation](https://ethoml.github.io/VAME/). <br/> <br/>
❗ <b>[Click here to read the open-access 2024 Resource article published in Cell Reports.](https://www.cell.com/cell-reports/fulltext/S2211-1247(24)01221-X) ❗ <br/>


We are a group of behavioral enthusiasts, comprising the original VAME developers Kevin Luxem and Pavol Bauer; behavioral neuroscientists Stephanie R. Miller, Jorge J. Palop; statisticians Alex Pico, Reuben Thomas; and machine learning researchers Pranav Nambiar, Katie Ly, Shreya Bangera, Dakota Mallen, Zhihao Cheng. Our aim is to provide scalable, unbiased and sensitive approaches for assessing mouse behavior using computer vision and machine learning approaches.

We are focused on the expanding the analytical capabilities of VAME segmentation by providing curated scripts for VAME implementation and tools for data processing, visualization, and statistical analyses.

## Recent Improvements to VAME
* Curated scripts for VAME implementation
* Addition of compatability with [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), [SLEAP](https://github.com/talmolab/sleap), [LightningPose](https://github.com/paninski-lab/lightning-pose), and [NWB](https://www.nwb.org/) files (via the [ndx-pose](https://github.com/rly/ndx-pose) extension)
* Addition of compatability with [movement](https://github.com/neuroinformatics-unit/movement) for data ingestion
* Addition of a new cost function for community dendrogram generation
* Addition of a new egocentric alignment method
* Refined output filename structure
  

## Authors and Code Contributors
VAME was developed by Kevin Luxem and Pavol Bauer (Luxem et. al., 2022). The original VAME repository was deprecated, forked, and is now being maintained here at https://github.com/EthoML/VAME.

The development of VAME is heavily inspired by [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/). As such, the VAME project management codebase has been adapted from the DeepLabCut codebase. The DeepLabCut 2.0 toolbox is © A. & M.W. Mathis Labs [deeplabcut.org](http:\\deeplabcut.org), released under LGPL v3.0. The implementation of the VRAE model is partially adapted from the [Timeseries clustering](https://github.com/tejaslodaya/timeseries-clustering-vae) repository developed by [Tejas Lodaya](https://tejaslodaya.com).

## VAME in a Nutshell

VAME is a framework to cluster behavioral signals obtained from pose-estimation tools. It is a [PyTorch](https://pytorch.org/)-based deep learning framework which leverages the power of recurrent neural networks (RNN) to model sequential data. In order to learn the underlying complex data distribution, we use the RNN in a variational autoencoder setting to extract the latent state of the animal in every step of the input time series.
The [step-by-step guide](https://ethoml.github.io/VAME/docs/getting_started/step_by_step/) walks through each step of the VAME workflow.

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

Follow the [step-by-step guide](https://ethoml.github.io/VAME/docs/getting_started/step_by_step/), or run the whole workflow with the [pipeline](https://ethoml.github.io/VAME/docs/getting_started/pipeline/). Both are also available as notebooks in the [examples folder](https://github.com/EthoML/VAME/tree/main/examples). To use VAME without writing code, see [VAME App](https://ethoml.github.io/VAME/docs/vame_app/).

VAME trains on a GPU when one is available (CUDA or Apple MPS) and falls back to the CPU otherwise.

## References
New 2024 Miller <i>et al</i>.: [Machine learning reveals prominent spontaneous behavioral changes and treatment efficacy in humanized and transgenic Alzheimer's disease models](https://www.cell.com/cell-reports/fulltext/S2211-1247(24)01221-X) <br/>
Original 2022 Luxem <i>et al</i>.: [Identifying Behavioral Structure from Deep Variational Embeddings of Animal Motion](https://www.biorxiv.org/content/10.1101/2020.05.14.095430v2) <br/>
See also: <br/>
Mocellin <i>et al</i>.: [A septal-ventral tegmental area circuit drives exploratory behavior](https://www.cell.com/neuron/fulltext/S0896-6273(23)00975-3) <br/>
Kingma & Welling: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) <br/>
Pereira & Silveira: [Learning Representations from Healthcare Time Series Data for Unsupervised Anomaly Detection](https://www.joao-pereira.pt/publications/accepted_version_BigComp19.pdf)

## License: GPLv3
See the [LICENSE file](https://github.com/EthoML/VAME/blob/main/LICENSE) for the full statement.
