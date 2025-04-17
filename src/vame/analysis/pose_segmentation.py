import os
import tqdm
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from hmmlearn import hmm
from sklearn.cluster import KMeans

from vame.schemas.states import save_state, SegmentSessionFunctionSchema
from vame.logging.logger import VameLogger, TqdmToLogger
from vame.model.rnn_model import RNN_VAE
from vame.io.load_poses import read_pose_estimation_file
from vame.util.cli import get_sessions_from_user_input
from vame.util.model_util import load_model
from vame.preprocessing.to_model import format_xarray_for_rnn


logger_config = VameLogger(__name__)
logger = logger_config.logger


def embedd_latent_vectors(
    config: dict,
    sessions: List[str],
    model: RNN_VAE,
    fixed: bool,
    read_from_variable: str = "position_processed",
    tqdm_stream: Union[TqdmToLogger, None] = None,
) -> List[np.ndarray]:
    """
    Embed latent vectors for the given files using the VAME model.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    sessions : List[str]
        List of session names.
    model : RNN_VAE
        VAME model.
    fixed : bool
        Whether the model is fixed.
    tqdm_stream : TqdmToLogger, optional
        TQDM Stream to redirect the tqdm output to logger.

    Returns
    -------
    List[np.ndarray]
        List of latent vectors for each file.
    """
    project_path = config["project_path"]
    temp_win = config["time_window"]
    num_features = config["num_features"]
    if not fixed:
        num_features = num_features - 3

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        pass
    else:
        torch.device("cpu")

    latent_vector_files = []

    for session in sessions:
        logger.info(f"Embedding of latent vector for file {session}")
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)
        data = np.copy(ds[read_from_variable].values)

        # Format the data for the RNN model
        data = format_xarray_for_rnn(
            ds=ds,
            read_from_variable=read_from_variable,
        )

        latent_vector_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(data.shape[1] - temp_win), file=tqdm_stream):
                # for i in tqdm.tqdm(range(10000)):
                data_sample_np = data[:, i : temp_win + i].T
                data_sample_np = np.reshape(data_sample_np, (1, temp_win, num_features))
                if use_gpu:
                    h_n = model.encoder(torch.from_numpy(data_sample_np).type("torch.FloatTensor").cuda())
                else:
                    h_n = model.encoder(torch.from_numpy(data_sample_np).type("torch.FloatTensor").to())
                mu, _, _ = model.lmbda(h_n)
                latent_vector_list.append(mu.cpu().data.numpy())

        latent_vector = np.concatenate(latent_vector_list, axis=0)
        latent_vector_files.append(latent_vector)

    return latent_vector_files


def get_latent_vectors(
    project_path: str,
    sessions: list,
    model_name: str,
    seg,
    n_clusters: int,
) -> List:
    """
    Gets all the latent vectors from each session into one list

    Parameters
    ----------
    project_path: str
        Path to vame project folder
    session: list
        List of sessions
    model_name: str
        Name of model
    seg: str
        Type of segmentation algorithm
    n_clusters : int
        Number of clusters.

    Returns
    -------
    List
        List of session latent vectors
    """

    latent_vectors = []  # list of session latent vectors
    for session in sessions:  # session loop to build latent_vector list
        latent_vector_path = os.path.join(
            str(project_path),
            "results",
            session,
            model_name,
            seg + "-" + str(n_clusters),
            "latent_vector_" + session + ".npy",
        )
        latent_vector = np.load(latent_vector_path)
        latent_vectors.append(latent_vector)
    return latent_vectors


def get_motif_usage(
    session_labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Count motif usage from session label array.

    Parameters
    ----------
    session_labels : np.ndarray
        Array of session labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Array of motif usage counts.
    """
    motif_usage = np.zeros(n_clusters)
    for i in range(n_clusters):
        motif_count = np.sum(session_labels == i)
        motif_usage[i] = motif_count
    # Include warning if any unused motifs are present
    unused_motifs = np.where(motif_usage == 0)[0]
    if unused_motifs.size > 0:
        logger.info(f"Warning: The following motifs are unused: {unused_motifs}")
    return motif_usage


def save_session_data(
    project_path: str,
    session: int,
    model_name: str,
    label: np.ndarray,
    cluster_center: np.ndarray,
    latent_vector: np.ndarray,
    motif_usage: np.ndarray,
    n_clusters: int,
    segmentation_algorithm: str,
):
    """
    Saves pose segmentation data for given session.

    Parameters
    ----------
    project_path: str
        Path to the vame project folder.
    session: int
        Session of interest to segment.
    model_name: str
        Name of model
    label: np.ndarray
        Array of the session's motif labels.
    cluster_center: np.ndarray
        Array of the session's kmeans cluster centers location in the latent space.
    latent_vector: np.ndarray,
        Array of the session's latent vectors.
    motif_usage: np.ndarray
        Array of the session's motif usage counts.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm: str
        Type of segmentation method, either 'kmeans or 'hmm'.

    Returns
    -------
    None
    """
    session_results_path = os.path.join(
        str(project_path),
        "results",
        session,
        model_name,
        segmentation_algorithm + "-" + str(n_clusters),
    )
    if not os.path.exists(session_results_path):
        try:
            os.mkdir(session_results_path)
        except OSError as error:
            logger.error(error)

    np.save(
        os.path.join(session_results_path, str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session),
        label,
    )
    if segmentation_algorithm == "kmeans":
        np.save(
            os.path.join(session_results_path, "cluster_center_" + session),
            cluster_center,
        )
    np.save(
        os.path.join(session_results_path, "latent_vector_" + session),
        latent_vector,
    )
    np.save(
        os.path.join(session_results_path, "motif_usage_" + session),
        motif_usage,
    )

    logger.info(f"Saved {session} segmentation data")


def same_segmentation(
    config: dict,
    sessions: List[str],
    latent_vectors: List[np.ndarray],
    n_clusters: int,
    segmentation_algorithm: str,
) -> None:
    """
    Apply the same segmentation to all animals.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    sessions : List[str]
        List of session names.
    latent_vectors : List[np.ndarray]
        List of latent vector arrays.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm : str
        Segmentation algorithm.

    Returns
    -------
    None
    """
    # List of arrays containing each session's motif labels #[SRM, 10/28/24], recommend rename this and similar variables to allsessions_labels
    labels = []  # List of array containing each session's motif labels
    cluster_center = []  # List of arrays containing each session's cluster centers
    motif_usages = []  # List of arrays containing each session's motif usages

    latent_vector_cat = np.concatenate(latent_vectors, axis=0)
    if segmentation_algorithm == "kmeans":
        logger.info("Using kmeans as segmentation algorithm!")
        kmeans = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
        ).fit(latent_vector_cat)
        cluster_center = kmeans.cluster_centers_
        # 1D, vector of all labels for the entire cohort
        label = kmeans.predict(latent_vector_cat)

    elif segmentation_algorithm == "hmm":
        if not config["hmm_trained"]:
            logger.info("Using a HMM as segmentation algorithm!")
            hmm_model = hmm.GaussianHMM(
                n_components=n_clusters,
                covariance_type="full",
                n_iter=100,
            )
            hmm_model.fit(latent_vector_cat)
            label = hmm_model.predict(latent_vector_cat)
            save_data = os.path.join(config["project_path"], "results", "")
            with open(save_data + "hmm_trained.pkl", "wb") as file:
                pickle.dump(hmm_model, file)
        else:
            logger.info("Using a pretrained HMM as segmentation algorithm!")
            save_data = os.path.join(config["project_path"], "results", "")
            with open(save_data + "hmm_trained.pkl", "rb") as file:
                hmm_model = pickle.load(file)
            label = hmm_model.predict(latent_vector_cat)

    idx = 0  # start index for each session
    for i, session in enumerate(sessions):
        file_len = latent_vectors[i].shape[0]  # stop index of the session
        session_labels = label[idx : idx + file_len]
        # labels.append(label[idx : idx + file_len])  # append session's label
        # if segmentation_algorithm == "kmeans":
        #     cluster_centers.append(cluster_center) #will this be the same for each session?

        # session's motif usage
        motif_usage = get_motif_usage(session_labels, n_clusters)
        motif_usages.append(motif_usage)
        idx += file_len  # updating the session start index

        save_session_data(
            config["project_path"],
            session,
            config["model_name"],
            session_labels,
            cluster_center,
            latent_vectors[i],
            motif_usage,
            n_clusters,
            segmentation_algorithm,
        )


def individual_segmentation(
    config: dict,
    sessions: List[str],
    latent_vectors: List[np.ndarray],
    n_clusters: int,
) -> Tuple:
    """
    Apply individual segmentation to each session.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    sessions : List[str]
        List of session names.
    latent_vectors : List[np.ndarray]
        List of latent vector arrays.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    Tuple
        Tuple of labels, cluster centers, and motif usages.
    """
    random_state = config["random_state_kmeans"]
    n_init = config["n_init_kmeans"]
    labels = []
    cluster_centers = []
    motif_usages = []
    for i, session in enumerate(sessions):
        logger.info(f"Processing session: {session}")
        kmeans = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        ).fit(latent_vectors[i])
        clust_center = kmeans.cluster_centers_
        label = kmeans.predict(latent_vectors[i])
        motif_usage = get_motif_usage(
            session_labels=label,
            n_clusters=n_clusters,
        )
        motif_usages.append(motif_usage)
        labels.append(label)
        cluster_centers.append(clust_center)

        save_session_data(
            config["project_path"],
            session,
            config["model_name"],
            labels[i],
            cluster_centers[i],
            latent_vectors[i],
            motif_usages[i],
            n_clusters,
            "kmeans",
        )
    return labels, cluster_centers, motif_usages


@save_state(model=SegmentSessionFunctionSchema)
def segment_session(
    config: dict,
    overwrite: bool = False,
    save_logs: bool = False,
) -> None:
    """
    Perform pose segmentation using the VAME model.
    Fills in the values in the "segment_session" key of the states.json file.
    Creates files at:
    - project_name/
        - results/
            - hmm_trained.pkl
            - session/
                - model_name/
                    - hmm-n_clusters/
                        - latent_vector_session.npy
                        - motif_usage_session.npy
                        - n_cluster_label_session.npy
                    - kmeans-n_clusters/
                        - latent_vector_session.npy
                        - motif_usage_session.npy
                        - n_cluster_label_session.npy
                        - cluster_center_session.npy

    latent_vector_session.npy contains the projection of the data into the latent space,
    for each frame of the video. Dimmentions: (n_frames, n_latent_features)

    motif_usage_session.npy contains the number of times each motif was used in the video.
    Dimmentions: (n_motifs,)

    n_cluster_label_session.npy contains the label of the cluster assigned to each frame.
    Dimmentions: (n_frames,)

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    overwrite : bool, optional
        Whether to overwrite existing segmentation results, by default False.
    save_logs : bool, optional
        Whether to save logs, by default False.

    Returns
    -------
    None
    """
    project_path = Path(config["project_path"]).resolve()
    try:
        tqdm_stream = None
        if save_logs:
            log_path = project_path / "logs" / "pose_segmentation.log"
            logger_config.add_file_handler(str(log_path))
            tqdm_stream = TqdmToLogger(logger)
        model_name = config["model_name"]
        n_clusters = config["n_clusters"]
        fixed = config["egocentric_data"]
        segmentation_algorithms = config["segmentation_algorithms"]
        ind_seg = config["individual_segmentation"]
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("Using CUDA")
            logger.info("GPU active: {}".format(torch.cuda.is_available()))
            logger.info("GPU used: {}".format(torch.cuda.get_device_name(0)))
        else:
            logger.info("CUDA is not working! Attempting to use the CPU...")
            torch.device("cpu")
        logger.info("Pose segmentation for VAME model: %s \n" % model_name)
        logger.info(f"Segmentation algorithms: {segmentation_algorithms}")

        for seg in segmentation_algorithms:
            logger.info("---------------------------------------------------------------------")

            # Get sessions to analyze
            sessions = []
            if config["all_data"] in ["Yes", "yes", "True", "true", True]:
                sessions = config["session_names"]
            else:
                sessions = get_sessions_from_user_input(
                    config=config,
                    action_message="run segmentation",
                )

            # Check if each session general results path exists
            for session in sessions:
                session_results_path = os.path.join(
                    str(project_path),
                    "results",
                    session,
                    model_name,
                )
                if not os.path.exists(session_results_path):
                    os.mkdir(session_results_path)

            # Checks if segment session was already processed before
            latent_vectors = []
            seg_results_path = os.path.join(
                str(project_path),
                "results",
                sessions[0],
                model_name,
                seg + "-" + str(n_clusters),
            )
            if os.path.exists(seg_results_path):
                if not overwrite:
                    logger.info(f"Segmentation for {seg} algorithm and cluster size {n_clusters} already exists, skipping...")
                    return
                logger.info(f"Segmentation for {seg} algorithm and cluster size {n_clusters} already exists, but will be overwritten.")
            else:
                logger.info(f"Starting segmentation for {seg} algorithm and cluster size {n_clusters}...")

            model = load_model(config, model_name, fixed)
            latent_vectors = embedd_latent_vectors(
                config=config,
                sessions=sessions,
                model=model,
                fixed=fixed,
                tqdm_stream=tqdm_stream,
            )

            # Apply same or indiv segmentation of latent vectors for each session
            if ind_seg:
                logger.info(f"Apply individual segmentation of latent vectors for each session, {n_clusters} clusters")
                labels, cluster_center, motif_usages = individual_segmentation(
                    config=config,
                    sessions=sessions,
                    latent_vectors=latent_vectors,
                    n_clusters=n_clusters,
                )
            else:
                logger.info(f"Apply the same segmentation of latent vectors for all sessions, {n_clusters} clusters")
                same_segmentation(
                    config=config,
                    sessions=sessions,
                    latent_vectors=latent_vectors,
                    n_clusters=n_clusters,
                    segmentation_algorithm=seg,
                )

            logger.info(
                "You succesfully extracted motifs with VAME! From here, you can proceed running vame.community() "
                "to get the full picture of the spatiotemporal dynamic. To get an idea of the behavior captured by VAME, "
                "run vame.motif_videos(). This will leave you with short snippets of certain movements."
            )

    except Exception as e:
        logger.exception(f"An error occurred during pose segmentation: {e}")
    finally:
        logger_config.remove_file_handler()
