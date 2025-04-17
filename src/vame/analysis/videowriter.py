import os
from pathlib import Path
import numpy as np
import cv2 as cv
import tqdm
from typing import Union, Optional, List
import imageio

from vame.util.auxiliary import read_config
from vame.util.cli import get_sessions_from_user_input
from vame.schemas.states import (
    save_state,
    MotifVideosFunctionSchema,
    CommunityVideosFunctionSchema,
)
from vame.logging.logger import VameLogger, TqdmToLogger
from vame.schemas.project import SegmentationAlgorithms


logger_config = VameLogger(__name__)
logger = logger_config.logger


def create_cluster_videos(
    config: dict,
    path_to_file: str,
    session: str,
    n_clusters: int,
    video_type: str,
    flag: str,
    segmentation_algorithm: SegmentationAlgorithms,
    output_video_type: str = ".mp4",
    tqdm_logger_stream: Union[TqdmToLogger, None] = None,
) -> None:
    """
    Generate cluster videos and save them to filesystem on project folder.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    path_to_file : str
        Path to the file.
    session : str
        Name of the session.
    n_clusters : int
        Number of clusters.
    video_type : str
        Type of input video.
    flag : str
        Flag indicating the type of video (motif or community).
    segmentation_algorithm : SegmentationAlgorithms
        Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'.
    output_video_type : str, optional
        Type of output video. Default is '.mp4'.
    tqdm_logger_stream : TqdmToLogger, optional
        Tqdm logger stream. Default is None.

    Returns
    -------
    None
    """
    if output_video_type not in [".mp4", ".avi"]:
        raise ValueError("Output video type must be either '.avi' or '.mp4'.")

    if flag == "motif":
        logger.info("Motif videos getting created for " + session + " ...")
        labels = np.load(
            os.path.join(
                path_to_file,
                str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
            )
        )
    if flag == "community":
        logger.info("Community videos getting created for " + session + " ...")
        labels = np.load(
            os.path.join(
                path_to_file,
                "community",
                "cohort_community_label_" + session + ".npy",
            )
        )

    video_file_path = os.path.join(
        config["project_path"],
        "data",
        "raw",
        session + video_type,
    )
    capture = cv.VideoCapture(video_file_path)
    if not capture.isOpened():
        raise ValueError(f"Video capture could not be opened. Ensure the video file is valid.\n {video_file_path}")
    width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = 25  # capture.get(cv.CAP_PROP_FPS)

    cluster_start = config["time_window"] / 2
    unique_labels, count_labels = np.unique(labels, return_counts=True)

    for cluster in unique_labels:
        logger.info("Cluster: %d" % (cluster))
        cluster_lbl = np.where(labels == cluster)
        cluster_lbl = cluster_lbl[0]
        if not cluster_lbl.size:
            logger.info("Cluster is empty")
            continue

        if flag == "motif":
            output = os.path.join(
                path_to_file,
                "cluster_videos",
                session + f"-motif_%d{output_video_type}" % cluster,
            )
        if flag == "community":
            output = os.path.join(
                path_to_file,
                "community_videos",
                session + f"-community_%d{output_video_type}" % cluster,
            )

        if output_video_type == ".avi":
            codec = cv.VideoWriter_fourcc("M", "J", "P", "G")
            video_writer = cv.VideoWriter(output, codec, fps, (int(width), int(height)))
        elif output_video_type == ".mp4":
            video_writer = imageio.get_writer(
                output,
                fps=fps,
                codec="h264",
                macro_block_size=None,
            )

        if len(cluster_lbl) < config["length_of_motif_video"]:
            vid_length = len(cluster_lbl)
        else:
            vid_length = config["length_of_motif_video"]

        for num in tqdm.tqdm(range(vid_length), file=tqdm_logger_stream):
            idx = cluster_lbl[num]
            capture.set(1, idx + cluster_start)
            ret, frame = capture.read()
            if output_video_type == ".avi":
                video_writer.write(frame)
            elif output_video_type == ".mp4":
                video_writer.append_data(frame)
        if output_video_type == ".avi":
            video_writer.release()
        elif output_video_type == ".mp4":
            video_writer.close()
    capture.release()


@save_state(model=MotifVideosFunctionSchema)
def motif_videos(
    config: dict,
    segmentation_algorithms: Optional[List[SegmentationAlgorithms]] = None,
    video_type: str = ".mp4",
    output_video_type: str = ".mp4",
    save_logs: bool = False,
) -> None:
    """
    Generate motif videos and save them to filesystem.
    Fills in the values in the "motif_videos" key of the states.json file.
    Files are saved at:
    - project_name/
        - results/
            - session_name/
                - model_name/
                    - segmentation_algorithm-n_clusters/
                        - cluster_videos/
                            - session_name-motif_0.mp4
                            - session_name-motif_1.mp4
                            - ...

    Parameters
    ----------
    config : dict
        Configuration parameters.
    segmentation_algorithm : SegmentationAlgorithms
        Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'.
        If None, it will be taken from the config file.
    video_type : str, optional
        Type of video. Default is '.mp4'.
    output_video_type : str, optional
        Type of output video. Default is '.mp4'.
    save_logs : bool, optional
        Save logs to filesystem. Default is False.

    Returns
    -------
    None
    """
    try:
        tqdm_logger_stream = None
        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "motif_videos.log"
            logger_config.add_file_handler(str(log_path))
            tqdm_logger_stream = TqdmToLogger(logger=logger)
        model_name = config["model_name"]
        n_clusters = config["n_clusters"]
        segmentation_algorithms_config = config["segmentation_algorithms"]
        if segmentation_algorithms is None:
            segmentation_algorithms = segmentation_algorithms_config
        else:
            for segmentation_algorithm in segmentation_algorithms:
                if segmentation_algorithm not in segmentation_algorithms_config:
                    raise ValueError(
                        f"Segmentation algorithm {segmentation_algorithm} was not used n project {config['project_name']}."
                    )

        # Get sessions
        if config["all_data"] in ["Yes", "yes", "True", "true", True]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                config=config,
                action_message="write motif videos",
            )

        for session in sessions:
            for segmentation_algorithm in segmentation_algorithms:
                logger.info(f"Creating motif videos for session {session}, algorithm: {segmentation_algorithm}, n_clusters: {n_clusters}")
                path_to_file = os.path.join(
                    config["project_path"],
                    "results",
                    session,
                    model_name,
                    segmentation_algorithm + "-" + str(n_clusters),
                    "",
                )
                if not os.path.exists(os.path.join(path_to_file, "cluster_videos")):
                    os.mkdir(os.path.join(path_to_file, "cluster_videos"))

                create_cluster_videos(
                    config=config,
                    path_to_file=path_to_file,
                    session=session,
                    n_clusters=n_clusters,
                    video_type=video_type,
                    flag="motif",
                    segmentation_algorithm=segmentation_algorithm,
                    output_video_type=output_video_type,
                    tqdm_logger_stream=tqdm_logger_stream,
                )
            logger.info("All videos have been created!")
    except Exception as e:
        logger.exception(f"Error in motif_videos: {e}")
        raise e
    finally:
        logger_config.remove_file_handler()


@save_state(model=CommunityVideosFunctionSchema)
def community_videos(
    config: dict,
    video_type: str = ".mp4",
    output_video_type: str = ".mp4",
    save_logs: bool = False,
) -> None:
    """
    Generate community videos and save them to filesystem on project community_videos folder.
    Fills in the values in the "community_videos" key of the states.json file.
    Files are saved at:
    - project_name/
        - results/
            - session_name/
                - model_name/
                    - segmentation_algorithm-n_clusters/
                        - community_videos/
                            - session_name-community_0.mp4
                            - session_name-community_1.mp4
                            - ...

    Parameters
    ----------
    config : dict
        Configuration parameters.
    segmentation_algorithm : SegmentationAlgorithms
        Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'.
    video_type : str, optional
        Type of video. Default is '.mp4'.
    save_logs : bool, optional
        Save logs to filesystem. Default is False.
    output_video_type : str, optional
        Type of output video. Default is '.mp4'.

    Returns
    -------
    None
    """
    try:
        tqdm_logger_stream = None

        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "community_videos.log"
            logger_config.add_file_handler(str(log_path))
            tqdm_logger_stream = TqdmToLogger(logger=logger)

        model_name = config["model_name"]
        n_clusters = config["n_clusters"]
        segmentation_algorithms = config["segmentation_algorithms"]

        # Get sessions
        if config["all_data"] in ["Yes", "yes", "True", "true", True]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                config=config,
                action_message="write community videos",
            )

        for session in sessions:
            for segmentation_algorithm in segmentation_algorithms:
                logger.info(f"Creating motif videos for session {session}, algorithm: {segmentation_algorithm}, n_clusters: {n_clusters}")
                path_to_file = os.path.join(
                    config["project_path"],
                    "results",
                    session,
                    model_name,
                    segmentation_algorithm + "-" + str(n_clusters),
                    "",
                )
                if not os.path.exists(os.path.join(path_to_file, "community_videos")):
                    os.mkdir(os.path.join(path_to_file, "community_videos"))

                create_cluster_videos(
                    config=config,
                    path_to_file=path_to_file,
                    session=session,
                    n_clusters=n_clusters,
                    video_type=video_type,
                    flag="community",
                    segmentation_algorithm=segmentation_algorithm,
                    tqdm_logger_stream=tqdm_logger_stream,
                    output_video_type=output_video_type,
                )

        logger.info("All videos have been created!")
    except Exception as e:
        logger.exception(f"Error in community_videos: {e}")
        raise e
    finally:
        logger_config.remove_file_handler()
