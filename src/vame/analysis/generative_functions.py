import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from vame.schemas.states import GenerativeModelFunctionSchema, save_state
from vame.logging.logger import VameLogger
from vame.util.auxiliary import read_config
from vame.util.model_util import load_model
from vame.util.cli import get_sessions_from_user_input
from vame.schemas.project import SegmentationAlgorithms


logger_config = VameLogger(__name__)
logger = logger_config.logger


def random_generative_samples_motif(
    config: dict,
    model: torch.nn.Module,
    latent_vector: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> plt.Figure:
    """
    Generate random samples for motifs.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    model : torch.nn.Module
        PyTorch model.
    latent_vector : np.ndarray
        Latent vectors.
    labels : np.ndarray
        Labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    plt.Figure
        Figure of generated samples.
    """
    logger.info("Generate random generative samples for motifs...")
    time_window = config["time_window"]
    for j in range(n_clusters):
        inds = np.where(labels == j)
        motif_latents = latent_vector[inds[0], :]
        gm = GaussianMixture(n_components=10).fit(motif_latents)

        # draw sample from GMM
        density_sample = gm.sample(10)

        # generate image via model decoder
        tensor_sample = torch.from_numpy(density_sample[0]).type("torch.FloatTensor")
        if torch.cuda.is_available():
            tensor_sample = tensor_sample.cuda()
        else:
            tensor_sample = tensor_sample.cpu()

        decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
        decoder_inputs = decoder_inputs.permute(0, 2, 1)

        image_sample = model.decoder(decoder_inputs, tensor_sample)
        recon_sample = image_sample.cpu().detach().numpy()

        fig, axs = plt.subplots(2, 5)
        for i in range(5):
            axs[0, i].plot(recon_sample[i, ...])
            axs[1, i].plot(recon_sample[i + 5, ...])
        plt.suptitle("Generated samples for motif " + str(j))
        return fig


def random_generative_samples(
    config: dict,
    model: torch.nn.Module,
    latent_vector: np.ndarray,
) -> plt.Figure:
    """
    Generate random generative samples.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    model : torch.nn.Module
        PyTorch model.
    latent_vector : np.ndarray
        Latent vectors.

    Returns
    -------
    plt.Figure
        Figure of generated samples.
    """
    logger.info("Generate random generative samples...")
    # Latent sampling and generative model
    time_window = config["time_window"]
    gm = GaussianMixture(n_components=10).fit(latent_vector)

    # draw sample from GMM
    density_sample = gm.sample(10)

    # generate image via model decoder
    tensor_sample = torch.from_numpy(density_sample[0]).type("torch.FloatTensor")
    if torch.cuda.is_available():
        tensor_sample = tensor_sample.cuda()
    else:
        tensor_sample = tensor_sample.cpu()

    decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
    decoder_inputs = decoder_inputs.permute(0, 2, 1)

    image_sample = model.decoder(decoder_inputs, tensor_sample)
    recon_sample = image_sample.cpu().detach().numpy()

    fig, axs = plt.subplots(2, 5)
    for i in range(5):
        axs[0, i].plot(recon_sample[i, ...])
        axs[1, i].plot(recon_sample[i + 5, ...])
    plt.suptitle("Generated samples")
    return fig


def random_reconstruction_samples(
    config: dict,
    model: torch.nn.Module,
    latent_vector: np.ndarray,
) -> plt.Figure:
    """
    Generate random reconstruction samples.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    model : torch.nn.Module
        PyTorch model to use.
    latent_vector : np.ndarray
        Latent vectors.

    Returns
    -------
    plt.Figure
        Figure of reconstructed samples.
    """
    logger.info("Generate random reconstruction samples...")
    # random samples for reconstruction
    time_window = config["time_window"]

    rnd = np.random.choice(latent_vector.shape[0], 10)
    tensor_sample = torch.from_numpy(latent_vector[rnd]).type("torch.FloatTensor")
    if torch.cuda.is_available():
        tensor_sample = tensor_sample.cuda()
    else:
        tensor_sample = tensor_sample.cpu()

    decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
    decoder_inputs = decoder_inputs.permute(0, 2, 1)

    image_sample = model.decoder(decoder_inputs, tensor_sample)
    recon_sample = image_sample.cpu().detach().numpy()

    fig, axs = plt.subplots(2, 5)
    for i in range(5):
        axs[0, i].plot(recon_sample[i, ...])
        axs[1, i].plot(recon_sample[i + 5, ...])
    plt.suptitle("Reconstructed samples")
    return fig


def visualize_cluster_center(
    config: dict,
    model: torch.nn.Module,
    cluster_center: np.ndarray,
) -> plt.Figure:
    """
    Visualize cluster centers.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    model : torch.nn.Module
        PyTorch model.
    cluster_center : np.ndarray
        Cluster centers.

    Returns
    -------
    plt.Figure
        Figure of cluster centers.
    """
    # Cluster Center
    logger.info("Visualize cluster center...")
    time_window = config["time_window"]
    animal_centers = cluster_center

    tensor_sample = torch.from_numpy(animal_centers).type("torch.FloatTensor")
    if torch.cuda.is_available():
        tensor_sample = tensor_sample.cuda()
    else:
        tensor_sample = tensor_sample.cpu()
    decoder_inputs = tensor_sample.unsqueeze(2).repeat(1, 1, time_window)
    decoder_inputs = decoder_inputs.permute(0, 2, 1)

    image_sample = model.decoder(decoder_inputs, tensor_sample)
    recon_sample = image_sample.cpu().detach().numpy()

    num = animal_centers.shape[0]
    b = int(np.ceil(num / 5))

    fig, axs = plt.subplots(5, b)
    idx = 0
    for k in range(5):
        for i in range(b):
            axs[k, i].plot(recon_sample[idx, ...])
            axs[k, i].set_title("Cluster %d" % idx)
            idx += 1
    return fig


@save_state(model=GenerativeModelFunctionSchema)
def generative_model(
    config: dict,
    segmentation_algorithm: SegmentationAlgorithms,
    mode: str = "sampling",
    save_logs: bool = False,
) -> plt.Figure:
    """
    Generative model.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    mode : str, optional
        Mode for generating samples. Defaults to "sampling".

    Returns
    -------
    plt.Figure
        Plots of generated samples for each segmentation algorithm.
    """
    try:
        if save_logs:
            logs_path = Path(config["project_path"]) / "logs" / "generative_model.log"
            logger_config.add_file_handler(str(logs_path))
        logger.info(f"Running generative model with mode {mode}...")
        model_name = config["model_name"]
        n_clusters = config["n_clusters"]

        # Get sessions
        if config["all_data"] in ["Yes", "yes"]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                config=config,
                action_message="generate samples",
            )

        model = load_model(config, model_name, fixed=False)

        for session in sessions:
            path_to_file = os.path.join(
                config["project_path"],
                "results",
                session,
                model_name,
                segmentation_algorithm + "-" + str(n_clusters),
                "",
            )

            if mode == "sampling":
                latent_vector = np.load(
                    os.path.join(
                        path_to_file,
                        "latent_vector_" + session + ".npy",
                    )
                )
                return random_generative_samples(
                    config,
                    model,
                    latent_vector,
                )

            if mode == "reconstruction":
                latent_vector = np.load(
                    os.path.join(
                        path_to_file,
                        "latent_vector_" + session + ".npy",
                    )
                )
                return random_reconstruction_samples(
                    config,
                    model,
                    latent_vector,
                )

            if mode == "centers":
                if segmentation_algorithm != "kmeans":
                    raise ValueError(
                        f"Algorithm {segmentation_algorithm} not supported for cluster center visualization."
                    )
                cluster_center = np.load(
                    os.path.join(
                        path_to_file,
                        "cluster_center_" + session + ".npy",
                    )
                )
                return visualize_cluster_center(
                    config,
                    model,
                    cluster_center,
                )

            if mode == "motifs":
                latent_vector = np.load(
                    os.path.join(
                        path_to_file,
                        "latent_vector_" + session + ".npy",
                    )
                )
                labels = np.load(
                    os.path.join(
                        path_to_file,
                        "",
                        str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
                    )
                )
                return random_generative_samples_motif(
                    config=config,
                    model=model,
                    latent_vector=latent_vector,
                    labels=labels,
                    n_clusters=n_clusters,
                )
    except Exception as e:
        logger.exception(str(e))
        raise
    finally:
        logger_config.remove_file_handler()
