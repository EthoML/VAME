---
sidebar_label: rnn_model
title: model.rnn_model
---

## Encoder Objects

```python
class Encoder(nn.Module)
```

Encoder module of the Variational Autoencoder.

#### \_\_init\_\_

```python
def __init__(NUM_FEATURES: int, hidden_size_layer_1: int,
             hidden_size_layer_2: int, dropout_encoder: float)
```

Initialize the Encoder module.

**Parameters**

* **NUM_FEATURES** (`int`): Number of input features.
* **hidden_size_layer_1** (`int`): Size of the first hidden layer.
* **hidden_size_layer_2** (`int`): Size of the second hidden layer.
* **dropout_encoder** (`float`): Dropout rate for regularization.

#### forward

```python
def forward(inputs: torch.Tensor) -> torch.Tensor
```

Forward pass of the Encoder module.

**Parameters**

* **inputs** (`torch.Tensor`): Input tensor of shape (batch_size, sequence_length, num_features).

**Returns**

* `torch.Tensor:`: Encoded representation tensor of shape (batch_size, hidden_size_layer_1 * 4).

## Lambda Objects

```python
class Lambda(nn.Module)
```

Lambda module for computing the latent space parameters.

#### \_\_init\_\_

```python
def __init__(ZDIMS: int, hidden_size_layer_1: int, softplus: bool)
```

Initialize the Lambda module.

**Parameters**

* **ZDIMS** (`int`): Size of the latent space.
* **hidden_size_layer_1** (`int`): Size of the first hidden layer.
* **hidden_size_layer_2** (`int, deprecated`): Size of the second hidden layer.
* **softplus** (`bool`): Whether to use softplus activation for logvar.

#### forward

```python
def forward(
        hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Forward pass of the Lambda module.

**Parameters**

* **hidden** (`torch.Tensor`): Hidden representation tensor of shape (batch_size, hidden_size_layer_1 * 4).

**Returns**

* `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Latent space tensor, mean tensor, logvar tensor.

## Decoder Objects

```python
class Decoder(nn.Module)
```

Decoder module of the Variational Autoencoder.

#### \_\_init\_\_

```python
def __init__(TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int,
             hidden_size_rec: int, dropout_rec: float)
```

Initialize the Decoder module.

**Parameters**

* **TEMPORAL_WINDOW** (`int`): Size of the temporal window.
* **ZDIMS** (`int`): Size of the latent space.
* **NUM_FEATURES** (`int`): Number of input features.
* **hidden_size_rec** (`int`): Size of the recurrent hidden layer.
* **dropout_rec** (`float`): Dropout rate for regularization.

#### forward

```python
def forward(inputs: torch.Tensor, z: torch.Tensor) -> torch.Tensor
```

Forward pass of the Decoder module.

**Parameters**

* **inputs** (`torch.Tensor`): Input tensor of shape (batch_size, seq_len, ZDIMS).
* **z** (`torch.Tensor`): Latent space tensor of shape (batch_size, ZDIMS).

**Returns**

* `torch.Tensor:`: Decoded output tensor of shape (batch_size, seq_len, NUM_FEATURES).

## Decoder\_Future Objects

```python
class Decoder_Future(nn.Module)
```

Decoder module for predicting future sequences.

#### \_\_init\_\_

```python
def __init__(TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int,
             FUTURE_STEPS: int, hidden_size_pred: int, dropout_pred: float)
```

Initialize the Decoder_Future module.

**Parameters**

* **TEMPORAL_WINDOW** (`int`): Size of the temporal window.
* **ZDIMS** (`int`): Size of the latent space.
* **NUM_FEATURES** (`int`): Number of input features.
* **FUTURE_STEPS** (`int`): Number of future steps to predict.
* **hidden_size_pred** (`int`): Size of the prediction hidden layer.
* **dropout_pred** (`float`): Dropout rate for regularization.

#### forward

```python
def forward(inputs: torch.Tensor, z: torch.Tensor) -> torch.Tensor
```

Forward pass of the Decoder_Future module.

**Parameters**

* **inputs** (`torch.Tensor`): Input tensor of shape (batch_size, seq_len, ZDIMS).
* **z** (`torch.Tensor`): Latent space tensor of shape (batch_size, ZDIMS).

**Returns**

* `torch.Tensor:`: Predicted future tensor of shape (batch_size, FUTURE_STEPS, NUM_FEATURES).

## RNN\_VAE Objects

```python
class RNN_VAE(nn.Module)
```

Variational Autoencoder module.

#### \_\_init\_\_

```python
def __init__(TEMPORAL_WINDOW: int, ZDIMS: int, NUM_FEATURES: int,
             FUTURE_DECODER: bool, FUTURE_STEPS: int, hidden_size_layer_1: int,
             hidden_size_layer_2: int, hidden_size_rec: int,
             hidden_size_pred: int, dropout_encoder: float, dropout_rec: float,
             dropout_pred: float, softplus: bool)
```

Initialize the VAE module.

**Parameters**

* **TEMPORAL_WINDOW** (`int`): Size of the temporal window.
* **ZDIMS** (`int`): Size of the latent space.
* **NUM_FEATURES** (`int`): Number of input features.
* **FUTURE_DECODER** (`bool`): Whether to include a future decoder.
* **FUTURE_STEPS** (`int`): Number of future steps to predict.
* **hidden_size_layer_1** (`int`): Size of the first hidden layer.
* **hidden_size_layer_2** (`int`): Size of the second hidden layer.
* **hidden_size_rec** (`int`): Size of the recurrent hidden layer.
* **hidden_size_pred** (`int`): Size of the prediction hidden layer.
* **dropout_encoder** (`float`): Dropout rate for encoder.

#### forward

```python
def forward(seq: torch.Tensor) -> tuple
```

Forward pass of the VAE.

**Parameters**

* **seq** (`torch.Tensor`): Input sequence tensor of shape (batch_size, seq_len, NUM_FEATURES).

**Returns**

* `Tuple containing:`: - If FUTURE_DECODER is True:
    - prediction (torch.Tensor): Reconstructed input sequence tensor.
    - future (torch.Tensor): Predicted future sequence tensor.
    - z (torch.Tensor): Latent representation tensor.
    - mu (torch.Tensor): Mean of the latent distribution tensor.
    - logvar (torch.Tensor): Log variance of the latent distribution tensor.
- If FUTURE_DECODER is False:
    - prediction (torch.Tensor): Reconstructed input sequence tensor.
    - z (torch.Tensor): Latent representation tensor.
    - mu (torch.Tensor): Mean of the latent distribution tensor.
    - logvar (torch.Tensor): Log variance of the latent distribution tensor.

