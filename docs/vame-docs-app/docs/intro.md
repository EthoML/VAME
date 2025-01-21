---
title: Introduction
sidebar_position: 1
---

🌟 Welcome to EthoML/VAME (Variational Animal Motion Encoding), an open-source machine learning tool for behavioral segmentation and analyses.

We are a group of behavioral enthusiasts, comprising the original VAME developers Kevin Luxem and Pavol Bauer, behavioral neuroscientists Stephanie R. Miller and Jorge J. Palop, and computer scientists and statisticians Alex Pico, Reuben Thomas, and Katie Ly). Our aim is to provide scalable, unbiased and sensitive approaches for assessing mouse behavior using computer vision and machine learning approaches.

We are focused on the expanding the analytical capabilities of VAME segmentation by providing curated scripts for VAME implementation and tools for data processing, visualization, and statistical analyses.

## VAME in a Nutshell

![VAME](/img/behavior_structure_crop.gif)

VAME is a framework to cluster behavioral signals obtained from pose-estimation tools. It is a [PyTorch](https://pytorch.org/)-based deep learning framework which leverages the power of recurrent neural networks (RNN) to model sequential data. In order to learn the underlying complex data distribution, we use the RNN in a variational autoencoder setting to extract the latent state of the animal in every step of the input time series.
The workflow of VAME consists of 5 steps and we explain them in detail [here](https://github.com/LINCellularNeuroscience/VAME/wiki/1.-VAME-Workflow)


## Authors and Code Contributors
VAME was developed by **Kevin Luxem** and **Pavol Bauer** (Luxem et. al., 2022). The original VAME repository was deprecated, forked, and is now being maintained here at https://github.com/EthoML/VAME.

The development of VAME is heavily inspired by [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/). As such, the VAME project management codebase has been adapted from the DeepLabCut codebase. The DeepLabCut 2.0 toolbox is © A. & M.W. Mathis Labs [deeplabcut.org](http:\\deeplabcut.org), released under LGPL v3.0. The implementation of the VRAE model is partially adapted from the [Timeseries clustering](https://github.com/tejaslodaya/timeseries-clustering-vae) repository developed by [Tejas Lodaya](https://tejaslodaya.com).

