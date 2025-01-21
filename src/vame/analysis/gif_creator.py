import os
import tqdm
import umap
import numpy as np
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple

from vame.util.auxiliary import read_config
from vame.util.gif_pose_helper import get_animal_frames
from vame.util.cli import get_sessions_from_user_input
from vame.logging.logger import VameLogger
from vame.schemas.project import SegmentationAlgorithms


logger_config = VameLogger(__name__)
logger = logger_config.logger


def create_video(
    path_to_file: str,
    session: str,
    embed: np.ndarray,
    clabel: np.ndarray,
    frames: List[np.ndarray],
    start: int,
    length: int,
    max_lag: int,
    num_points: int,
) -> None:
    """
    Create video frames for the given embedding.

    Parameters
    ----------
    path_to_file : str
        Path to the file.
    session : str
        Session name.
    embed : np.ndarray
        Embedding array.
    clabel : np.ndarray
        Cluster labels.
    frames : List[np.ndarray]
        List of frames.
    start : int
        Starting index.
    length : int
        Length of the video.
    max_lag : int
        Maximum lag.
    num_points : int
        Number of points.

    Returns
    -------
    None
    """
    # set matplotlib colormap
    cmap = matplotlib.cm.gray
    cmap_reversed = plt.get_cmap("gray_r")

    # this here generates every frame for your gif. The gif is lastly created by using ImageJ
    # the embed variable is my umap embedding, which is for the 2D case a 2xn dimensional vector
    fig = plt.figure()
    spec = GridSpec(ncols=2, nrows=1, width_ratios=[6, 3])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])
    ax2.axis("off")
    ax2.grid(False)
    lag = 0
    for i in tqdm.tqdm(range(length)):
        if i > max_lag:
            lag = i - max_lag
        ax1.cla()
        ax1.axis("off")
        ax1.grid(False)
        if clabel is not None:
            ax1.scatter(
                embed[:num_points, 0],
                embed[:num_points, 1],
                c=clabel[:num_points],
                cmap="Spectral",
                s=1,
                alpha=0.4,
            )
        else:
            ax1.scatter(embed[:num_points, 0], embed[:num_points, 1], s=1, alpha=0.4)

        ax1.set_aspect("equal", "datalim")
        ax1.plot(
            embed[start + lag : start + i, 0],
            embed[start + lag : start + i, 1],
            ".b-",
            alpha=0.6,
            linewidth=2,
            markersize=4,
        )
        ax1.plot(embed[start + i, 0], embed[start + i, 1], "gx", markersize=4)
        frame = frames[i]
        ax2.imshow(frame, cmap=cmap_reversed)
        # ax2.set_title("Motif %d,\n Community: %s" % (lbl, motifs[lbl]), fontsize=10)
        fig.savefig(os.path.join(path_to_file, "gif_frames", session + "gif_%d.png") % i)


def gif(
    config: str,
    pose_ref_index: list,
    segmentation_algorithm: SegmentationAlgorithms,
    subtract_background: bool = True,
    start: int | None = None,
    length: int = 500,
    max_lag: int = 30,
    label: str = "community",
    file_format: str = ".mp4",
    crop_size: Tuple[int, int] = (300, 300),
) -> None:
    """Create a GIF from the given configuration.

    Parameters
    ----------
    config : str
        Path to the configuration file.
    pose_ref_index : list
        List of reference coordinate indices for alignment.
    segmentation_algorithm : SegmentationAlgorithms
        Segmentation algorithm.
    subtract_background : bool, optional
        Whether to subtract background. Defaults to True.
    start :int, optional
        Starting index. Defaults to None.
    length : int, optional
        Length of the video. Defaults to 500.
    max_lag : int, optional
        Maximum lag. Defaults to 30.
    label : str, optional
        Label type [None, community, motif]. Defaults to 'community'.
    file_format : str, optional
        File format. Defaults to '.mp4'.
    crop_size : Tuple[int, int], optional
        Crop size. Defaults to (300,300).

    Returns
    -------
    None
    """
    config_file = Path(config).resolve()
    cfg = read_config(str(config_file))
    model_name = cfg["model_name"]
    n_clusters = cfg["n_clusters"]

    if segmentation_algorithm not in cfg["segmentation_algorithms"]:
        raise ValueError("Segmentation algorithm not found in config")

    # Get sessions
    if cfg["all_data"] in ["Yes", "yes"]:
        sessions = cfg["session_names"]
    else:
        sessions = get_sessions_from_user_input(
            cfg=cfg,
            action_message="create gifs",
        )

    for session in sessions:
        path_to_file = os.path.join(
            cfg["project_path"],
            "results",
            session,
            model_name,
            segmentation_algorithm + "-" + str(n_clusters),
            "",
        )
        if not os.path.exists(os.path.join(path_to_file, "gif_frames")):
            os.mkdir(os.path.join(path_to_file, "gif_frames"))

        embed = np.load(
            os.path.join(
                path_to_file,
                "community",
                "umap_embedding_" + session + ".npy",
            )
        )

        try:
            embed = np.load(
                os.path.join(
                    path_to_file,
                    "community",
                    "umap_embedding_" + session + ".npy",
                )
            )
            num_points = cfg["num_points"]
            if num_points > embed.shape[0]:
                num_points = embed.shape[0]
        except Exception:
            logger.info(f"Compute embedding for session {session}")
            reducer = umap.UMAP(
                n_components=2,
                min_dist=cfg["min_dist"],
                n_neighbors=cfg["n_neighbors"],
                random_state=cfg["random_state"],
            )

            latent_vector = np.load(os.path.join(path_to_file, "", "latent_vector_" + session + ".npy"))

            num_points = cfg["num_points"]
            if num_points > latent_vector.shape[0]:
                num_points = latent_vector.shape[0]
            logger.info("Embedding %d data points.." % num_points)

            embed = reducer.fit_transform(latent_vector[:num_points, :])
            np.save(
                os.path.join(
                    path_to_file,
                    "community",
                    "umap_embedding_" + session + ".npy",
                ),
                embed,
            )

        if label == "motif":
            umap_label = np.load(
                os.path.join(
                    path_to_file,
                    str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
                )
            )
        elif label == "community":
            umap_label = np.load(
                os.path.join(
                    path_to_file,
                    "community",
                    "cohort_community_label_" + session + ".npy",
                )
            )
        elif label is None:
            umap_label = None

        if start is None:
            start = np.random.choice(embed[:num_points].shape[0] - length)
        else:
            start = start

        frames = get_animal_frames(
            cfg,
            session,
            pose_ref_index,
            start,
            length,
            subtract_background,
            file_format,
            crop_size,
        )
        create_video(
            path_to_file,
            session,
            embed,
            umap_label,
            frames,
            start,
            length,
            max_lag,
            num_points,
        )
