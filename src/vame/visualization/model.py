import os
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.utils.data as Data
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from vame.model.rnn_vae import RNN_VAE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    data = x[:, :seq_len_half, :].type("torch.FloatTensor").to(DEVICE)
    data_fut = x[:, seq_len_half : seq_len_half + FUTURE_STEPS, :].type("torch.FloatTensor").to(DEVICE)

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
        fig.savefig(os.path.join(filepath, "evaluate", "future_reconstruction.png"))
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
        plt.close()
    else:
        plt.close(fig)


def plot_loss(
    config: dict,
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
                - mse_and_kl_loss_model_name.png

    Parameters
    ----------
    config : dict
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
    basepath = os.path.join(config["project_path"], "model", config["testing_name"], "model_losses")
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
        evaluate_path = os.path.join(config["project_path"], "model", config["testing_name"], "evaluate", "")
        fig.savefig(evaluate_path + "Loss_plot_" + model_name + ".png")

    if show_figure:
        plt.show()
        plt.close()
    else:
        plt.close(fig)


def visualize_latent_space(
        config: dict,
        model: RNN_VAE,
        dataloader: Data.DataLoader,
        FUTURE_DECODER,
        TEMPORAL_WINDOW,
        CLUSTERS,
        save_to_file: bool,
        results_folder: Optional[str] = None
):
    latent_vectors = []
    seq_len_half = int(TEMPORAL_WINDOW / 2)

    with torch.no_grad():
        for data in dataloader:
            data = data.permute(0, 2, 1)
            data = data[:, :seq_len_half, :]
            data = data.type("torch.FloatTensor").to(DEVICE)
            if FUTURE_DECODER:
                data_tilde, future, latent, mu, logvar = model(data)
            else:
                data_tilde, latent, mu, logvar = model(data)

            z = model.reparameterize(mu, logvar)
            
            latent_vectors.append(z.cpu().numpy())
            # can use reconstruction loss as labels
    
    # Flatten layers 
    latent_vectors = np.concatenate(latent_vectors)

    # Clustering
    kmeans = kmeans = KMeans(
            init="k-means++",
            n_clusters=CLUSTERS,
            random_state=42,
            n_init=20,
        ).fit(latent_vectors)
    labels = kmeans.fit_predict(latent_vectors)

    fig, ax = plt.subplots(1,2, figsize = (13,7))
    # Reduce dimensionality
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    ax[0].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                          c=labels, cmap='viridis', s=1, alpha = 0.3) 
    ax[0].set_title('VAE Latent Space (PCA)')

    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)

    # Plot
    scatter = ax[1].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                          c=labels, cmap='viridis', s=1, alpha = 0.3) 
    # ax[1].colorbar(scatter)
    ax[1].set_title('VAE Latent Space (tsne)')
    ax[1].set_xlabel('t-SNE Dimension 1')
    ax[1].set_ylabel('t-SNE Dimension 2')

    if save_to_file:
        evaluate_path = os.path.join(config["project_path"], "model", config["testing_name"], "evaluate", "")
        fig.savefig(evaluate_path + "Latent_space_PCA_tsne_" + config["model_name"] + ".png")

    plt.show()
    plt.close()