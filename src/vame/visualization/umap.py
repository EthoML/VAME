import os
import umap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional

from vame.util.cli import get_sessions_from_user_input
from vame.schemas.states import VisualizeUmapFunctionSchema, save_state
from vame.logging.logger import VameLogger
from vame.schemas.project import SegmentationAlgorithms


logger_config = VameLogger(__name__)
logger = logger_config.logger


def umap_embedding(
    config: dict,
    session: str,
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: SegmentationAlgorithms,
) -> np.ndarray:
    """
    Perform UMAP embedding for given file and parameters.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    session : str
        Session name.
    model_name : str
        Model name.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm : str
        Segmentation algorithm.

    Returns
    -------
    np.ndarray
        UMAP embedding.
    """
    reducer = umap.UMAP(
        n_components=2,
        min_dist=config["min_dist"],
        n_neighbors=config["n_neighbors"],
        random_state=config["random_state"],
    )
    logger.info(f"UMAP calculation for session {session}")
    folder = os.path.join(
        config["project_path"],
        "results",
        session,
        model_name,
        segmentation_algorithm + "-" + str(n_clusters),
        "",
    )
    latent_vector = np.load(os.path.join(folder, "latent_vector_" + session + ".npy"))
    num_points = config["num_points"]
    if num_points > latent_vector.shape[0]:
        num_points = latent_vector.shape[0]
    logger.info(f"Embedding {num_points} data points...")
    embed = reducer.fit_transform(latent_vector[:num_points, :])
    np.save(
        os.path.join(folder, "community", "umap_embedding_" + session + ".npy"),
        embed,
    )
    return embed


def umap_vis(
    embed: np.ndarray,
    num_points: int,
    labels: Optional[np.ndarray] = None,
) -> Figure:
    """
    Visualize UMAP embedding without labels.

    Parameters
    ----------
    embed : np.ndarray
        UMAP embedding.
    num_points : int
        Number of data points to visualize.
    labels : np.ndarray, optional
        Motif or community labels. Default is None.

    Returns
    -------
    plt.Figure
        Plot Visualization of UMAP embedding.
    """
    scatter_kwargs = {
        "x": embed[:num_points, 0],
        "y": embed[:num_points, 1],
        "s": 2,
        "alpha": 0.5,
    }
    if labels is not None:
        scatter_kwargs["c"] = labels[:num_points]
        scatter_kwargs["cmap"] = "Spectral"
        scatter_kwargs["alpha"] = 0.7

    plt.close("all")
    fig = plt.figure(1)
    plt.scatter(**scatter_kwargs)
    plt.gca().set_aspect("equal", "datalim")
    plt.grid(False)
    return fig


@save_state(model=VisualizeUmapFunctionSchema)
def visualize_umap(
    config: dict,
    label: Optional[str] = None,
    save_logs: bool = False,
) -> None:
    """
    Visualize UMAP embeddings based on configuration settings.
    Fills in the values in the "visualization_umap" key of the states.json file.
    Saves results files at:
    - project_name/
        - results/
            - file_name/
                - model_name/
                    - segmentation_algorithm-n_clusters/
                        - community/
                            - umap_embedding_file_name.npy
                            - umap_vis_label_none_file_name.png  (UMAP visualization without labels)
                            - umap_vis_motif_file_name.png  (UMAP visualization with motif labels)
                            - umap_vis_community_file_name.png  (UMAP visualization with community labels)

    Parameters
    ----------
    config : dict
        Configuration parameters.
    label : str, optional
        Type of labels to visualize. Options are None, 'motif' or 'community'. Default is None.
    save_logs : bool, optional
        Save logs to file. Default is False.

    Returns
    -------
    None
    """
    try:
        if save_logs:
            logs_path = Path(config["project_path"]) / "logs" / "visualization.log"
            logger_config.add_file_handler(str(logs_path))

        model_name = config["model_name"]
        n_clusters = config["n_clusters"]
        segmentation_algorithms = config["segmentation_algorithms"]

        # Get sessions
        if config["all_data"] in ["Yes", "yes"]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                config=config,
                action_message="generate visualization",
            )

        for session in sessions:
            for seg in segmentation_algorithms:
                base_path = Path(config["project_path"]) / "results" / session / model_name / f"{seg}-{n_clusters}"
                umap_embeddings_path = base_path / "community" / f"umap_embedding_{session}.npy"
                if umap_embeddings_path.exists():
                    logger.info(f"UMAP embedding already exists for session {session}")
                    embed = np.load(str(umap_embeddings_path.resolve()))
                else:
                    logger.info(f"Computing UMAP embedding for session {session}")
                    if not (base_path / "community").exists():
                        os.mkdir(os.path.join(base_path, "community"))
                    embed = umap_embedding(
                        config=config,
                        session=session,
                        model_name=model_name,
                        n_clusters=n_clusters,
                        segmentation_algorithm=seg,
                    )

                num_points = config["num_points"]
                if num_points > embed.shape[0]:
                    num_points = embed.shape[0]

                if label is None:
                    output_figure_file_name = "umap_vis_label_none_" + session + ".png"
                    labels = None
                elif label == "motif":
                    output_figure_file_name = "umap_vis_motif_" + session + ".png"
                    labels_file_path = base_path / f"{n_clusters}_{seg}_label_{session}.npy"
                    labels = np.load(str(labels_file_path.resolve()))
                elif label == "community":
                    output_figure_file_name = "umap_vis_community_" + session + ".png"
                    labels_file_path = base_path / "community" / f"cohort_community_label_{session}.npy"
                    labels = np.load(str(labels_file_path.resolve()))

                output_figure = umap_vis(
                    embed=embed,
                    num_points=num_points,
                    labels=labels,
                )
                fig_path = base_path / "community" / output_figure_file_name
                output_figure.savefig(fig_path)
                logger.info(f"UMAP figure saved to {fig_path}")

    except Exception as e:
        logger.exception(str(e))
        raise e
    finally:
        logger_config.remove_file_handler()
