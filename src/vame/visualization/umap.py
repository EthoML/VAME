import os
import umap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

from vame.util.cli import get_sessions_from_user_input
from vame.schemas.states import VisualizeUmapFunctionSchema, save_state
from vame.logging.logger import VameLogger
from vame.schemas.project import SegmentationAlgorithms


logger_config = VameLogger(__name__)
logger = logger_config.logger


def umap_embedding(
    cfg: dict,
    session: str,
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: SegmentationAlgorithms,
) -> np.ndarray:
    """
    Perform UMAP embedding for given file and parameters.

    Parameters
    ----------
    cfg : dict
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
        min_dist=cfg["min_dist"],
        n_neighbors=cfg["n_neighbors"],
        random_state=cfg["random_state"],
    )
    logger.info(f"UMAP calculation for session {session}")
    folder = os.path.join(
        cfg["project_path"],
        "results",
        session,
        model_name,
        segmentation_algorithm + "-" + str(n_clusters),
        "",
    )
    latent_vector = np.load(os.path.join(folder, "latent_vector_" + session + ".npy"))
    num_points = cfg["num_points"]
    if num_points > latent_vector.shape[0]:
        num_points = latent_vector.shape[0]
    logger.info(f"Embedding {num_points} data points...")
    embed = reducer.fit_transform(latent_vector[:num_points, :])
    np.save(
        os.path.join(folder, "community", "umap_embedding_" + session + ".npy"),
        embed,
    )
    return embed


# def umap_vis_community_labels(cfg: dict, embed: np.ndarray, community_labels_all: np.ndarray, save_path: str | None) -> None:
#     """Create plotly visualizaton of UMAP embedding with community labels.

#     Args:
#         cfg (dict): Configuration parameters.
#         embed (np.ndarray): UMAP embedding.
#         community_labels_all (np.ndarray): Community labels.
#         save_path: Path to save the plot. If None it will not save the plot.

#     Returns
#         None
#     """
#     num_points = cfg['num_points']
#     community_labels_all = np.asarray(community_labels_all)
#     if num_points > community_labels_all.shape[0]:
#         num_points = community_labels_all.shape[0]
#     logger.info("Embedding %d data points.." %num_points)

#     num = np.unique(community_labels_all)

#     fig = plt.figure(1)
#     plt.scatter(
#         embed[:,0],
#         embed[:,1],
#         c=community_labels_all[:num_points],
#         cmap='Spectral',
#         s=2,
#         alpha=1
#     )
#     plt.colorbar(boundaries=np.arange(np.max(num)+2)-0.5).set_ticks(np.arange(np.max(num)+1))
#     plt.gca().set_aspect('equal', 'datalim')
#     plt.grid(False)

#     if save_path is not None:
#         plt.savefig(save_path)
#         return fig
#     plt.show()
#     return fig


def umap_vis(
    embed: np.ndarray,
    num_points: int,
) -> plt.Figure:
    """
    Visualize UMAP embedding without labels.

    Parameters
    ----------
    embed : np.ndarray
        UMAP embedding.
    num_points : int
        Number of data points to visualize.

    Returns
    -------
    plt.Figure
        Plot Visualization of UMAP embedding.
    """
    # plt.cla()
    # plt.clf()
    plt.close("all")
    fig = plt.figure(1)
    plt.scatter(embed[:num_points, 0], embed[:num_points, 1], s=2, alpha=0.5)
    plt.gca().set_aspect("equal", "datalim")
    plt.grid(False)
    return fig


def umap_label_vis(
    embed: np.ndarray,
    label: np.ndarray,
    num_points: int,
) -> plt.Figure:
    """
    Visualize UMAP embedding with motif labels.

    Parameters
    ----------
    embed : np.ndarray
        UMAP embedding.
    label : np.ndarray
        Motif labels.
    num_points : int
        Number of data points to visualize.

    Returns
    -------
    plt.Figure
        Plot figure of UMAP visualization embedding with motif labels.
    """
    fig = plt.figure(1)
    plt.scatter(
        embed[:num_points, 0],
        embed[:num_points, 1],
        c=label[:num_points],
        cmap="Spectral",
        s=2,
        alpha=0.7,
    )
    # plt.colorbar(boundaries=np.arange(n_clusters+1)-0.5).set_ticks(np.arange(n_clusters))
    plt.gca().set_aspect("equal", "datalim")
    plt.grid(False)
    return fig


def umap_vis_comm(
    embed: np.ndarray,
    community_label: np.ndarray,
    num_points: int,
) -> plt.Figure:
    """
    Visualize UMAP embedding with community labels.

    Parameters
    ----------
    embed : np.ndarray
        UMAP embedding.
    community_label : np.ndarray
        Community labels.
    num_points : int
        Number of data points to visualize.

    Returns
    -------
    plt.Figure
        Plot figure of UMAP visualization embedding with community labels.
    """
    num = np.unique(community_label).shape[0]
    fig = plt.figure(1)
    plt.scatter(
        embed[:num_points, 0],
        embed[:num_points, 1],
        c=community_label[:num_points],
        cmap="Spectral",
        s=2,
        alpha=0.7,
    )
    # plt.colorbar(boundaries=np.arange(num+1)-0.5).set_ticks(np.arange(num))
    plt.gca().set_aspect("equal", "datalim")
    plt.grid(False)
    return fig


@save_state(model=VisualizeUmapFunctionSchema)
def visualize_umap(
    config: dict,
    segmentation_algorithm: SegmentationAlgorithms,
    label: Optional[str] = None,
    save_logs: bool = False,
) -> None:
    """
    Visualize UMAP embeddings based on configuration settings.
    Fills in the values in the "visualization" key of the states.json file.
    Saves results files at:

    If label is None (UMAP visualization without labels):
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
    segmentation_algorithm : SegmentationAlgorithms
        Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'.
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

        # Get sessions
        if config["all_data"] in ["Yes", "yes"]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                cfg=config,
                action_message="generate visualization",
            )

        for idx, session in enumerate(sessions):
            path_to_file = os.path.join(
                config["project_path"],
                "results",
                session,
                "",
                model_name,
                "",
                segmentation_algorithm + "-" + str(n_clusters),
            )

            try:
                embed = np.load(
                    os.path.join(
                        path_to_file,
                        "",
                        "community",
                        "",
                        "umap_embedding_" + session + ".npy",
                    )
                )
                num_points = config["num_points"]
                if num_points > embed.shape[0]:
                    num_points = embed.shape[0]
            except Exception:
                if not os.path.exists(os.path.join(path_to_file, "community")):
                    os.mkdir(os.path.join(path_to_file, "community"))
                logger.info(f"Compute embedding for session {session}")
                embed = umap_embedding(
                    config,
                    session,
                    model_name,
                    n_clusters,
                    segmentation_algorithm,
                )
                num_points = config["num_points"]
                if num_points > embed.shape[0]:
                    num_points = embed.shape[0]

            if label is None:
                output_figure = umap_vis(embed, num_points)
                fig_path = os.path.join(
                    path_to_file,
                    "community",
                    "umap_vis_label_none_" + session + ".png",
                )
                output_figure.savefig(fig_path)

            if label == "motif":
                motif_label = np.load(
                    os.path.join(
                        path_to_file,
                        "",
                        str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
                    )
                )
                output_figure = umap_label_vis(
                    embed,
                    motif_label,
                    num_points,
                )
                fig_path = os.path.join(
                    path_to_file,
                    "community",
                    "umap_vis_motif_" + session + ".png",
                )
                output_figure.savefig(fig_path)

            if label == "community":
                community_label = np.load(
                    os.path.join(
                        path_to_file,
                        "",
                        "community",
                        "",
                        "cohort_community_label_" + session + ".npy",
                    )
                )
                output_figure = umap_vis_comm(embed, community_label, num_points)
                fig_path = os.path.join(
                    path_to_file,
                    "community",
                    "umap_vis_community_" + session + ".png",
                )
                output_figure.savefig(fig_path)
    except Exception as e:
        logger.exception(str(e))
        raise e
    finally:
        logger_config.remove_file_handler()
