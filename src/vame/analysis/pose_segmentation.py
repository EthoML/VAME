import os
import tqdm
import torch
import pickle
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Union
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from vame.schemas.states import save_state, SegmentSessionFunctionSchema
from vame.logging.logger import VameLogger, TqdmToLogger
from vame.model.rnn_model import RNN_VAE
from vame.io.load_poses import read_pose_estimation_file
from vame.util.cli import get_sessions_from_user_input
from vame.util.model_util import load_model
from vame.preprocessing.to_model import format_xarray_for_rnn
import math

logger_config = VameLogger(__name__)
logger = logger_config.logger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                h_n = model.encoder(torch.from_numpy(data_sample_np).type("torch.FloatTensor").to(DEVICE))
                mu, _, _ = model.lmbda(h_n)
                latent_vector_list.append(mu.cpu().data.numpy())

        latent_vector = np.concatenate(latent_vector_list, axis=0)
        latent_vector_files.append(latent_vector)

    return latent_vector_files

def get_latent_vectors(
        project_path: str,
        testing_name: str,
        sessions: list,
        seg, 
        n_clusters: int,
) -> List:
    """
    Gets all the latent vectors from each session into one list

    Parameters
    ----------
    project_path: str
        Path to vame project folder
    sessions: list
        List of sessions
    seg: str
        Type of segmentation algorithm
    n_clusters : int
        Number of clusters.

    Returns
    -------
    List
        List of session latent vectors
    """

    latent_vectors = [] #list of session latent vectors
    for session in sessions: #session loop to build latent_vector list
        latent_vector_path = os.path.join(
            str(project_path),
            "results",
            session,
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

def get_geometric_median(
    latent_vectors: np.ndarray,
) -> np.ndarray:
    centroid = np.mean(latent_vectors, axis=0)
    result = minimize(lambda c: np.sum(cdist([c], latent_vectors)), centroid)

    return result.x

def get_session_cluster_center(
    project_path: str,
    testing_name: str,
    session: int,
    segmentation_algorithm: str,
    latent_vector: np.ndarray,
    session_labels: np.ndarray,
    n_clusters: int,
    # center_type: str, #'centroid', 'geometric_mean'
    # vis_type: str, #'pca', 't-sne', 'umap'
) -> Tuple:
    
    session_cluster_centers_centroid = []
    session_cluster_centers_geo_median = []
    for i in range(0,n_clusters):
        # motif_center = np.empty([])
        motif_indices = list(np.where(session_labels == i)[0])
        motif_latent_vector = latent_vector[motif_indices]
        print(f'motif {i} latent_vector shape',motif_latent_vector.shape)

        motif_center_centroid = np.mean(motif_latent_vector, axis =0)
        motif_center_geo_median = get_geometric_median(motif_latent_vector)
        session_cluster_centers_centroid.append(motif_center_centroid)
        session_cluster_centers_geo_median.append(motif_center_geo_median)
        print('motif_center shape', motif_center_centroid.shape)
        print('motif_center shape', motif_center_geo_median.shape)

        #add session center distances to the model center

        # if center_type == 'centroid':
        #     motif_center = np.mean(motif_latent_vector, axis =0)
        # elif center_type == 'geometric_mean':
        #     motif_center= get_geometric_median(motif_latent_vector)
        # session_cluster_centers.append(motif_center)
        # print('motif_center shape', motif_center.shape)

        # if vis_type == 'pca':
        #     pca = PCA(n_components=2)
        #     transform_latent_vectors= pca.fit_transform(motif_latent_vector)
        #     transform_center = pca.transform(motif_center.reshape(1, -1))

        #     plt.clf()
        #     plt.scatter(transform_latent_vectors[:, 0], transform_latent_vectors[:, 1], alpha=0.1, label="Latent Points")
        #     plt.scatter(transform_center[:, 0], transform_center[:, 1], color='red', marker='x', label=center_type)
        #     plt.xlabel("PC1")
        #     plt.ylabel("PC2")
        #     # plt.title(f"{vis_type} Projection of Latent Vectors for Motif{i}, {center_type} center")
        #     # plt.legend()
        # if vis_type == 'umap':
        #     umap_reducer = umap.UMAP(n_components=2, random_state=42)
        #     reduced_points = umap_reducer.fit_transform(motif_latent_vector)
        #     reduced_center = umap_reducer.transform(motif_center.reshape(1, -1))

        #     plt.scatter(reduced_points[:, 0], reduced_points[:, 1], alpha=0.5, label="Cluster Points")
        #     plt.scatter(reduced_center[:, 0], reduced_center[:, 1], color='red', marker='x', label=center_type)
        
        # plt.title(f"{vis_type} Projection of Latent Vectors for Motif{i}, {center_type} center")
        # plt.legend()
        # figure_path = os.path.join(
        #                         project_path,
        #                         "results",
        #                         session,
        #                         model_name,
        #                         segmentation_algorithm + "-" + str(n_clusters),
        #                     )
        # plt.savefig(os.path.join(
        #             figure_path,
        #             'motif' + str(i) + center_type + '_cluster_center_' + vis_type + '_figure.png'
        # ))
    session_cluster_centers_centroid = np.array(session_cluster_centers_centroid)
    session_cluster_centers_geo_median = np.array(session_cluster_centers_geo_median)
    print('sesion_cluster_center shape', session_cluster_centers_centroid.shape)
    print('sesion_cluster_center shape', session_cluster_centers_geo_median.shape)

    return session_cluster_centers_centroid, session_cluster_centers_geo_median


from vame.analysis.pose_segmentation import get_latent_vectors
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.cm as cm
import random

def get_session_data(
    project_path: str,
    session: str,
    testing_name: str,
    seg: str,
    n_clusters: int
) -> tuple:
    
    results_path = os.path.join(project_path, 'results', testing_name ,session, seg + '-' + str(n_clusters))
    motif_labels_path = os.path.join(results_path,
                              str(n_clusters) + '_' + seg + '_label_' + session + '.npy')
    model_cluster_center_path = os.path.join(results_path,
                                             'cluster_center_model.npy')
    centroid_cluster_center_path = os.path.join(results_path, 
                                                'centroid_cluster_center_' + session + '.npy' )   
    geo_med_clsuter_center_path =  os.path.join(results_path, 
                                                'geo_median_cluster_center_' + session + '.npy' )

    session_motif_labels = np.load(motif_labels_path)
    model_cluster_center = np.load(model_cluster_center_path)
    centroid_cluster_center = np.load(centroid_cluster_center_path)
    geo_med_cluster_center = np.load(geo_med_clsuter_center_path)


    return session_motif_labels, model_cluster_center, centroid_cluster_center, geo_med_cluster_center

# duck: move out
def matrix_dimension(n_subplots):
    """
    Calculates the optimal plot dimensions (rows, cols) for a subplot grid with concideration for edge cases.
    * developed to handel varying numbers of n motifs

    Parameters
    ----------
    n_subplots : int
       total number of subplots needed

    Returns
    -------
    tuple (int,int)
        (row, col)
    """
    #initialize variables
    best_rows = 1
    best_cols = n_subplots
    min_diff = n_subplots - 1

    # check relavent divisors
    for rows in range(1, int(math.sqrt(n_subplots)) + 1):
        if n_subplots % rows == 0:
            cols = n_subplots // rows
            diff = abs(cols - rows)

            # check fit
            if diff < min_diff:
                min_diff = diff
                best_rows = rows
                best_cols = cols
            # Prioritize completely filled grids
            elif diff == min_diff and rows * cols == n_subplots:
                best_rows, best_cols = rows, cols

    
    if best_rows * best_cols != n_subplots:
        return None  # No valid dimensions found

    return (best_rows, best_cols)

def plot_cluster_centers(config: dict) -> None:
    project_path = config["project_path"]
    sessions = config["session_names"]
    # sessions = ['Session01']
    seg = 'kmeans'
    n_clusters = config["n_clusters"]
    plt_dims = matrix_dimension(n_clusters)

    # Get latent vectors for each session
    session_latent_vector = get_latent_vectors(project_path, sessions, seg, n_clusters)
    cohort_latent_vector = np.concatenate(session_latent_vector)  # Shape: (all frames, n_clusters)


    # PCA over the entire cohort
    pca = PCA(n_components=2)
    pca_latent_vectors = pca.fit_transform(cohort_latent_vector)

    # Color for each session
    cmap = cm.get_cmap('tab20b')  # 
    # cmap = ['red', 'darkorange', 'darkgreen', 'mediumblue', 'purple']

    cohort_pca_latent_vectors = []
    start_idx = 0

    fig, ax = plt.subplots(*plt_dims, figsize = (12,10), sharex = True, sharey= True)
    
    cur_motif = 0
    for row in range(plt_dims[0]):
        for col in range(plt_dims[1]):
            for i, session in enumerate(sessions):
                frames = session_latent_vector[i].shape[0]

                # Extract PCA-transformed session data
                session_pca = pca_latent_vectors[start_idx: start_idx + frames]
                cohort_pca_latent_vectors.append(session_pca)  # Append whole session PCA data
                start_idx += frames  # Update index for the next session
                # print('session pca', session_pca.shape)
                # print('start idx',start_idx)

                # Get session data
                s_motif_labels, model_cluster_center, centroid_cluster_center, geo_med_cluster_center = get_session_data(
                    project_path, session,testing_name, seg, n_clusters
                )
                motif_indices = np.where(s_motif_labels == cur_motif)[0]  # Get motif indices for this cluster
                # print(motif_indices.shape)
                motif_pca = session_pca[motif_indices]

                # Transform centers using PCA
                model_center_pca = pca.transform(model_cluster_center[cur_motif,:].reshape(1,-1))
                centroid_center_pca = pca.transform(centroid_cluster_center[cur_motif,:].reshape(1,-1)) #double check dimensions [m,:] or [:,m]
                geo_med_center_pca = pca.transform(geo_med_cluster_center[cur_motif,:].reshape(1,-1))

                # Scatter plot with a unique session color
                points = 100 if len(motif_pca) > 100 else len(motif_pca)
                random_list = random.sample(range(len(motif_pca)), points )
                ax[row][col].scatter(
                    motif_pca[random_list, 0], motif_pca[random_list, 1], 
                    # motif_pca[0:points, 0], motif_pca[0:points, 1], 
                    # motif_pca[:, 0], motif_pca[:, 1], 
                    alpha=0.1, color=cmap(i), s=3, label=f"{session} - Motif {cur_motif}")
                ax[row][col].scatter(centroid_center_pca[:, 0], centroid_center_pca[:, 1], 
                        color= cmap(i),s=50, marker='x', label=f"{session} Centroid Center")
                ax[row][col].scatter(geo_med_center_pca[:, 0], geo_med_center_pca[:, 1], 
                        color= cmap(i), s=50, marker='+', label=f"{session} Geo-Median Center")
                ax[row][col].set(title = 'Motif ' + str(cur_motif))
            
            start_idx = 0

            # # Plot model center
            ax[row][col].scatter(model_center_pca[:, 0], model_center_pca[:, 1], 
                        color='black', marker='X', label="Model Center")

            cur_motif += 1

    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle(f"PCA Projection of Latent Vectors for Motifs")
    plt.tight_layout()
    plt.show()

# plot_cluster_centers(config_dict)

def save_session_data(
    project_path: str,
    session: int,
    testing_name: str,
    label: np.ndarray,
    cluster_center: np.ndarray,
    session_cluster_center_centroid: np.ndarray,
    session_cluster_center_geo_median: np.ndarray,
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
                                testing_name,
                                session,
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
            os.path.join(session_results_path, "cluster_center_model"),
            cluster_center,
        )
        np.save(
            os.path.join(session_results_path, "centroid_cluster_center_" + session),
            session_cluster_center_centroid,
        )
        np.save(
            os.path.join(session_results_path, "geo_median_cluster_center_" + session),
            session_cluster_center_geo_median,
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
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
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
    Tuple
        Tuple of labels, cluster centers, and motif usages.
    """
    # List of arrays containing each session's motif labels #[SRM, 10/28/24], recommend rename this and similar variables to allsessions_labels
    labels = [] #List of array containing each session's motif labels
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
            save_data = os.path.join(config["project_path"], "results", config["testing_name"], "")
            with open(save_data + "hmm_trained.pkl", "wb") as file:
                pickle.dump(hmm_model, file)
        else:
            logger.info("Using a pretrained HMM as segmentation algorithm!")
            save_data = os.path.join(config["project_path"], "results", config["testing_name"], "")
            with open(save_data + "hmm_trained.pkl", "rb") as file:
                hmm_model = pickle.load(file)
            cohort_labels = hmm_model.predict(latent_vector_cat)# 1D, vector of all labels for the entire cohort

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
        center_type = 'geometric_mean'
        vis_type = 'pca'
        session_cluster_center_centroid, session_cluster_center_geo_median = get_session_cluster_center(config["project_path"],
                                                                                                        session,
                                                                                                        segmentation_algorithm,
                                                                                                        latent_vectors[i],
                                                                                                        session_labels,
                                                                                                        n_clusters,
                                                                                                        )

        #pass session_cluster_center to this function to save
        #make sure the dimensions are correct.
        save_session_data(config["project_path"], 
                          session, 
                          session_labels, 
                          cluster_center,
                          session_cluster_center_centroid,
                          session_cluster_center_geo_median,
                          latent_vectors[i],
                          motif_usage,
                          n_clusters,
                          segmentation_algorithm)

        # save_session_data(
        #     config["project_path"], 
        #     config["testing_name"],
        #     session, 
        #     session_labels, 
        #     cluster_center,
        #     latent_vectors[i],
        #     motif_usage,
        #     n_clusters,
        #     segmentation_algorithm
        # )



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

        save_session_data(config["project_path"], 
                          config["testing_name"],
                          session,
                          labels[i], 
                          cluster_centers[i],
                          latent_vectors[i],
                          motif_usages[i],
                          n_clusters,
                          'kmeans'
        )
    return labels, cluster_centers, motif_usages


@save_state(model=SegmentSessionFunctionSchema)
def segment_session(
    config: dict,
    save_logs: bool = False,
) -> None:
    """
    Perform pose segmentation using the VAME model.
    Fills in the values in the "segment_session" key of the states.json file.
    Creates files at:
    - project_name/
        - results/
            - hmm_trained.pkl
            - testing_name/ [optional]
                - session/
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
        testing_name = config['testing_name']
        n_clusters = config["n_clusters"]
        fixed = config["egocentric_data"]
        segmentation_algorithms = config["segmentation_algorithms"]
        ind_seg = config["individual_segmentation"]

        if torch.cuda.is_available():
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
            logger.info(f"Running pose segmentation using {seg} algorithm...")
            
            # Get sessions to analyze
            sessions = []
            if config["all_data"] in ["Yes", "yes"]:
                sessions = config["session_names"]
            else:
                sessions = get_sessions_from_user_input(
                    config=config,
                    action_message="run segmentation",
                )

            #Check if each session general results path exists
            for session in sessions:
                session_results_path = os.path.join(
                                            str(project_path),
                                            "results",
                                            testing_name,
                                            session,
                                        ) 
                if not os.path.exists(session_results_path):
                    os.makedirs(session_results_path)

            #PART 1:Determine to embedd or get latent vectors
            latent_vectors = []
            if not os.path.exists(
                    os.path.join(
                        str(project_path),
                        "results",
                        testing_name,
                        sessions[0], 
                        seg + "-" + str(n_clusters),
                    )
            ): #Checks if segment session was already processed before
                new_segmentation = True
                model = load_model(config, model_name, fixed, config['testing_name'])
                latent_vectors = embedd_latent_vectors( 
                    config,
                    sessions,
                    model,
                    fixed,
                    tqdm_stream=tqdm_stream,
                )

            else: #else results session[0] path exists
                logger.info(f"\nSegmentation with {n_clusters} k-means clusters already exists for model {model_name}")

                flag = input(
                    "WARNING: A segmentation for the chosen model and cluster size already exists! \n"
                    "Do you want to continue? A new segmentation will be computed! (yes/no) "
                )

                if flag == "yes": 
                    new_segmentation = True 
                    latent_vectors = get_latent_vectors(
                        project_path,
                        testing_name, 
                        sessions, 
                        seg, 
                        n_clusters
                    )

                else:
                    logger.info("No new segmentation has been calculated.")
                    new_segmentation = False

            #PART 2: Apply same or indiv segmentation of latent vectors for each session
            if ind_seg:
                logger.info(
                    f"Apply individual segmentation of latent vectors for each session, {n_clusters} clusters"
                )
                labels, cluster_center, motif_usages = individual_segmentation(
                    config=config,
                    sessions=sessions,
                    latent_vectors=latent_vectors,
                    n_clusters=n_clusters,
                )
            else:
                logger.info(
                    f"Apply the same segmentation of latent vectors for all sessions, {n_clusters} clusters"
                )
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
