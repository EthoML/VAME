---
sidebar_label: rnn_vae
title: model.rnn_vae
---

#### logger\_config

#### logger

#### tqdm\_to\_logger

#### use\_gpu

#### reconstruction\_loss

```python
def reconstruction_loss(x: torch.Tensor, x_tilde: torch.Tensor,
                        reduction: str) -> torch.Tensor
```

Compute the reconstruction loss between input and reconstructed data.

**Parameters**

* **x** (`torch.Tensor`): Input data tensor.
* **x_tilde** (`torch.Tensor`): Reconstructed data tensor.
* **reduction** (`str`): Type of reduction for the loss.

**Returns**

* `torch.Tensor`: Reconstruction loss.

#### future\_reconstruction\_loss

```python
def future_reconstruction_loss(x: torch.Tensor, x_tilde: torch.Tensor,
                               reduction: str) -> torch.Tensor
```

Compute the future reconstruction loss between input and predicted future data.

**Parameters**

* **x** (`torch.Tensor`): Input future data tensor.
* **x_tilde** (`torch.Tensor`): Reconstructed future data tensor.
* **reduction** (`str`): Type of reduction for the loss.

**Returns**

* `torch.Tensor`: Future reconstruction loss.

#### cluster\_loss

```python
def cluster_loss(H: torch.Tensor, kloss: int, lmbda: float,
                 batch_size: int) -> torch.Tensor
```

Compute the cluster loss.

**Parameters**

* **H** (`torch.Tensor`): Latent representation tensor.
* **kloss** (`int`): Number of clusters.
* **lmbda** (`float`): Lambda value for the loss.
* **batch_size** (`int`): Size of the batch.

**Returns**

* `torch.Tensor`: Cluster loss.

#### kullback\_leibler\_loss

```python
def kullback_leibler_loss(mu: torch.Tensor,
                          logvar: torch.Tensor) -> torch.Tensor
```

Compute the Kullback-Leibler divergence loss.
See Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 - https://arxiv.org/abs/1312.6114

Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

**Parameters**

* **mu** (`torch.Tensor`): Mean of the latent distribution.
* **logvar** (`torch.Tensor`): Log variance of the latent distribution.

**Returns**

* `torch.Tensor`: Kullback-Leibler divergence loss.

#### kl\_annealing

```python
def kl_annealing(epoch: int, kl_start: int, annealtime: int,
                 function: str) -> float
```

Anneal the Kullback-Leibler loss to let the model learn first the reconstruction of the data
before the KL loss term gets introduced.

**Parameters**

* **epoch** (`int`): Current epoch number.
* **kl_start** (`int`): Epoch number to start annealing the loss.
* **annealtime** (`int`): Annealing time.
* **function** (`str`): Annealing function type.

**Returns**

* `float`: Annealed weight value for the loss.

#### gaussian

```python
def gaussian(ins: torch.Tensor,
             is_training: bool,
             seq_len: int,
             std_n: float = 0.8) -> torch.Tensor
```

Add Gaussian noise to the input data.

**Parameters**

* **ins** (`torch.Tensor`): Input data tensor.
* **is_training** (`bool`): Whether it is training mode.
* **seq_len** (`int`): Length of the sequence.
* **std_n** (`float`): Standard deviation for the Gaussian noise.

**Returns**

* `torch.Tensor`: Noisy input data tensor.

#### train

```python
def train(train_loader: Data.DataLoader, epoch: int, model: nn.Module,
          optimizer: torch.optim.Optimizer, anneal_function: str, BETA: float,
          kl_start: int, annealtime: int, seq_len: int, future_decoder: bool,
          future_steps: int, scheduler: torch.optim.lr_scheduler._LRScheduler,
          mse_red: str, mse_pred: str, kloss: int, klmbda: float, bsize: int,
          noise: bool) -> Tuple[float, float, float, float, float, float]
```

Train the model.

**Parameters**

* **train_loader** (`DataLoader`): Training data loader.
* **epoch** (`int`): Current epoch number.
* **model** (`nn.Module`): Model to be trained.
* **optimizer** (`Optimizer`): Optimizer for training.
* **anneal_function** (`str`): Annealing function type.
* **BETA** (`float`): Beta value for the loss.
* **kl_start** (`int`): Epoch number to start annealing the loss.
* **annealtime** (`int`): Annealing time.
* **seq_len** (`int`): Length of the sequence.
* **future_decoder** (`bool`): Whether a future decoder is used.
* **future_steps** (`int`): Number of future steps to predict.
* **scheduler** (`lr_scheduler._LRScheduler`): Learning rate scheduler.
* **mse_red** (`str`): Reduction type for MSE reconstruction loss.
* **mse_pred** (`str`): Reduction type for MSE prediction loss.
* **kloss** (`int`): Number of clusters for cluster loss.
* **klmbda** (`float`): Lambda value for cluster loss.
* **bsize** (`int`): Size of the batch.
* **noise** (`bool`): Whether to add Gaussian noise to the input.

**Returns**

* `Tuple[float, float, float, float, float, float]`: Kullback-Leibler weight, train loss, K-means loss, KL loss,
MSE loss, future loss.

#### test

```python
def test(test_loader: Data.DataLoader, model: nn.Module, BETA: float,
         kl_weight: float, seq_len: int, mse_red: str, kloss: str,
         klmbda: float, future_decoder: bool,
         bsize: int) -> Tuple[float, float, float]
```

Evaluate the model on the test dataset.

**Parameters**

* **test_loader** (`DataLoader`): DataLoader for the test dataset.
* **model** (`nn.Module`): The trained model.
* **BETA** (`float`): Beta value for the VAE loss.
* **kl_weight** (`float`): Weighting factor for the KL divergence loss.
* **seq_len** (`int`): Length of the sequence.
* **mse_red** (`str`): Reduction method for the MSE loss.
* **kloss** (`str`): Loss function for K-means clustering.
* **klmbda** (`float`): Lambda value for K-means loss.
* **future_decoder** (`bool`): Flag indicating whether to use a future decoder.
* **bsize :int**: Batch size.

**Returns**

* `Tuple[float, float, float]`: Tuple containing MSE loss per item, total test loss per item,
and K-means loss weighted by the kl_weight.

#### train\_model

```python
@save_state(model=TrainModelFunctionSchema)
def train_model(config: dict, save_logs: bool = False) -> None
```

Train Variational Autoencoder using the configuration file values.
Fills in the values in the &quot;train_model&quot; key of the states.json file.
Creates files at:
- project_name/
    - model/
        - best_model/
            - snapshots/
                - model_name_Project_epoch_0.pkl
                - ...
            - model_name_Project.pkl
        - model_losses/
            - fut_losses_VAME.npy
            - kl_losses_VAME.npy
            - kmeans_losses_VAME.npy
            - mse_test_losses_VAME.npy
            - mse_train_losses_VAME.npy
            - test_losses_VAME.npy
            - train_losses_VAME.npy
            - weight_values_VAME.npy
        - pretrained_model/

**Parameters**

* **config** (`dict`): Configuration dictionary.
* **save_logs** (`bool, optional`): Whether to save the logs, by default False.

**Returns**

* `None`

