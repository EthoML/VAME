import os
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.utils.data as Data

from vame.model.rnn_vae import RNN_VAE


use_gpu = torch.cuda.is_available()
if use_gpu:
    pass
else:
    torch.device("cpu")


def plot_reconstruction(
    filepath: str,
    test_loader: Data.DataLoader,
    seq_len_half: int,
    model: RNN_VAE,
    model_name: str,
    FUTURE_DECODER: bool,
    FUTURE_STEPS: int,
    suffix: Optional[str] = None,
    show_figure: bool = False,
) -> None:
    """
    Plot the reconstruction and future prediction of the input sequence.
    Saves the plot to:
    - project_name/
        - model/
            - evaluate/
                - Reconstruction_model_name.png

    Parameters
    ----------
    filepath : str
        Path to save the plot.
    test_loader : Data.DataLoader
        DataLoader for the test dataset.
    seq_len_half : int
        Half of the temporal window size.
    model : RNN_VAE
        Trained VAE model.
    model_name : str
        Name of the model.
    FUTURE_DECODER : bool
        Flag indicating whether the model has a future prediction decoder.
    FUTURE_STEPS : int
        Number of future steps to predict.
    suffix : str, optional
        Suffix for the saved plot filename. Defaults to None.
    show_figure : bool, optional
        Flag indicating whether to show the plot. Defaults to False.

    Returns
    -------
    None
    """
    # x = test_loader.__iter__().next()
    dataiter = iter(test_loader)
    x = next(dataiter)
    x = x.permute(0, 2, 1)
    if use_gpu:
        data = x[:, :seq_len_half, :].type("torch.FloatTensor").cuda()
        data_fut = x[:, seq_len_half : seq_len_half + FUTURE_STEPS, :].type("torch.FloatTensor").cuda()
    else:
        data = x[:, :seq_len_half, :].type("torch.FloatTensor").to()
        data_fut = x[:, seq_len_half : seq_len_half + FUTURE_STEPS, :].type("torch.FloatTensor").to()
    if FUTURE_DECODER:
        x_tilde, future, latent, mu, logvar = model(data)

        fut_orig = data_fut.cpu()
        fut_orig = fut_orig.data.numpy()
        fut = future.cpu()
        fut = fut.detach().numpy()

    else:
        x_tilde, latent, mu, logvar = model(data)

    data_orig = data.cpu()
    data_orig = data_orig.data.numpy()
    data_tilde = x_tilde.cpu()
    data_tilde = data_tilde.detach().numpy()

    if FUTURE_DECODER:
        fig, axs = plt.subplots(2, 5)
        fig.suptitle("Reconstruction [top] and future prediction [bottom] of input sequence")
        for i in range(5):
            axs[0, i].plot(data_orig[i, ...], color="k", label="Sequence Data")
            axs[0, i].plot(
                data_tilde[i, ...],
                color="r",
                linestyle="dashed",
                label="Sequence Reconstruction",
            )
            axs[1, i].plot(fut_orig[i, ...], color="k")
            axs[1, i].plot(fut[i, ...], color="r", linestyle="dashed")
        axs[0, 0].set(xlabel="time steps", ylabel="reconstruction")
        axs[1, 0].set(xlabel="time steps", ylabel="predction")
        fig.savefig(os.path.join(filepath, "evaluate", "Future_Reconstruction.png"))
    else:
        fig, ax1 = plt.subplots(1, 5)
        for i in range(5):
            fig.suptitle("Reconstruction of input sequence")
            ax1[i].plot(data_orig[i, ...], color="k", label="Sequence Data")
            ax1[i].plot(
                data_tilde[i, ...],
                color="r",
                linestyle="dashed",
                label="Sequence Reconstruction",
            )
        fig.tight_layout()
        if not suffix:
            fig.savefig(
                os.path.join(filepath, "evaluate", "Reconstruction_" + model_name + ".png"),
                bbox_inches="tight",
            )
        elif suffix:
            fig.savefig(
                os.path.join(
                    filepath,
                    "evaluate",
                    "Reconstruction_" + model_name + "_" + suffix + ".png",
                ),
                bbox_inches="tight",
            )

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def plot_loss(
    cfg: dict,
    model_name: str,
    save_to_file: bool = False,
    show_figure: bool = True,
) -> None:
    """
    Plot the losses of the trained model.
    Saves the plot to:
    - project_name/
        - model/
            - evaluate/
                - MSE-and-KL-Loss_model_name.png

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    model_name : str
        Name of the model.
    save_to_file : bool, optional
        Flag indicating whether to save the plot. Defaults to False.
    show_figure : bool, optional
        Flag indicating whether to show the plot. Defaults to True.

    Returns
    -------
    None
    """
    basepath = os.path.join(cfg["project_path"], "model", "model_losses")
    train_loss = np.load(os.path.join(basepath, "train_losses_" + model_name + ".npy"))
    test_loss = np.load(os.path.join(basepath, "test_losses_" + model_name + ".npy"))
    mse_loss_train = np.load(os.path.join(basepath, "mse_train_losses_" + model_name + ".npy"))
    mse_loss_test = np.load(os.path.join(basepath, "mse_test_losses_" + model_name + ".npy"))
    km_losses = np.load(os.path.join(basepath, "kmeans_losses_" + model_name + ".npy"))
    kl_loss = np.load(os.path.join(basepath, "kl_losses_" + model_name + ".npy"))
    fut_loss = np.load(os.path.join(basepath, "fut_losses_" + model_name + ".npy"))

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle(f"Losses of model: {model_name}")
    ax1.set(xlabel="Epochs", ylabel="loss [log-scale]")
    ax1.set_yscale("log")
    ax1.plot(train_loss, label="Train-Loss")
    ax1.plot(test_loss, label="Test-Loss")
    ax1.plot(mse_loss_train, label="MSE-Train-Loss")
    ax1.plot(mse_loss_test, label="MSE-Test-Loss")
    ax1.plot(km_losses, label="KMeans-Loss")
    ax1.plot(kl_loss, label="KL-Loss")
    ax1.plot(fut_loss, label="Prediction-Loss")
    ax1.legend()

    if save_to_file:
        evaluate_path = os.path.join(cfg["project_path"], "model", "evaluate")
        fig.savefig(os.path.join(evaluate_path, "MSE-and-KL-Loss" + model_name + ".png"))

    if show_figure:
        plt.show()
    else:
        plt.close(fig)
