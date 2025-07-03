import os
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Literal

from vame.analysis.tree_hierarchy import (
    graph_to_tree,
    bag_nodes_by_cutline,
)
from vame.util.cli import get_sessions_from_user_input
from vame.visualization.community import draw_tree
from vame.schemas.states import save_state, CommunityFunctionSchema
from vame.schemas.project import SegmentationAlgorithms
from vame.logging.logger import VameLogger
from vame.analysis.pose_segmentation import get_motif_usage


logger_config = VameLogger(__name__)
logger = logger_config.logger


def get_adjacency_matrix(
    labels: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the adjacency matrix, transition matrix, and temporal matrix.

    Parameters
    ----------
    labels : np.ndarray
        Array of cluster labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing: adjacency matrix, transition matrix, and temporal matrix.
    """
    temp_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    adjacency_matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    cntMat = np.zeros((n_clusters))
    steps = len(labels)

    for i in range(n_clusters):
        for k in range(steps - 1):
            idx = labels[k]
            if idx == i:
                idx2 = labels[k + 1]
                if idx == idx2:
                    continue
                else:
                    cntMat[idx2] = cntMat[idx2] + 1
        temp_matrix[i] = cntMat
        cntMat = np.zeros((n_clusters))

    for k in range(steps - 1):
        idx = labels[k]
        idx2 = labels[k + 1]
        if idx == idx2:
            continue
        adjacency_matrix[idx, idx2] = 1
        adjacency_matrix[idx2, idx] = 1

    transition_matrix = get_transition_matrix(temp_matrix)
    return adjacency_matrix, transition_matrix, temp_matrix


def get_transition_matrix(
    adjacency_matrix: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Compute the transition matrix from the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix.
    threshold : float, optional
        Threshold for considering transitions. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        Transition matrix.
    """
    row_sum = adjacency_matrix.sum(axis=1)
    transition_matrix = adjacency_matrix / row_sum[:, np.newaxis]
    transition_matrix[transition_matrix <= threshold] = 0
    if np.any(np.isnan(transition_matrix)):
        transition_matrix = np.nan_to_num(transition_matrix)
    return transition_matrix


def get_motif_labels(
    config: dict,
    sessions: List[str],
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get motif labels and motif counts for the entire cohort.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    sessions : List[str]
        List of session names.
    model_name : str
        Model name.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm : str
        Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'.

    Returns
    -------
    Tuple [np.ndarray, np.ndarray]
        Tuple with:
            - Array of motif labels (integers) of the entire cohort
            - Array of motif counts of the entire cohort
    """
    # TODO  - this is limiting the number of frames to the minimum number of frames in all files
    # Is this intended behavior? and why?
    shapes = []
    for session in sessions:
        path_to_dir = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            segmentation_algorithm + "-" + str(n_clusters),
            "",
        )
        file_labels = np.load(
            os.path.join(
                path_to_dir,
                str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
            )
        )
        shape = len(file_labels)
        shapes.append(shape)
    shapes = np.array(shapes)

    cohort_motif_labels = []
    for session in sessions:
        path_to_dir = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            segmentation_algorithm + "-" + str(n_clusters),
            "",
        )
        file_labels = np.load(
            os.path.join(
                path_to_dir,
                str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
            )
        )
        cohort_motif_labels.extend(
            file_labels
        )  # add each element to community_label for example [1,2,3] instead of [1, [2,3]] #RENAME TO MOTIF_LABELS
    cohort_motif_labels = np.array(cohort_motif_labels)
    cohort_motif_counts = get_motif_usage(cohort_motif_labels, n_clusters)

    return cohort_motif_labels, cohort_motif_counts


def compute_transition_matrices(
    files: List[str],
    labels: List[np.ndarray],
    n_clusters: int,
) -> List[np.ndarray]:
    """
    Compute transition matrices for given files and labels.

    Parameters
    ----------
    files : List[str]
        List of file paths.
    labels : List[np.ndarray]
        List of label arrays.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    List[np.ndarray]:
        List of transition matrices.
    """
    transition_matrices = []
    for i, file in enumerate(files):
        adj, trans, mat = get_adjacency_matrix(labels[i], n_clusters)
        transition_matrices.append(trans)
    return transition_matrices


def create_cohort_community_bag(
    config: dict,
    motif_labels: List[np.ndarray],
    trans_mat_full: np.ndarray,
    cut_tree: int | None,
    n_clusters: int,
    segmentation_algorithm: Literal["hmm", "kmeans"],
) -> list:
    """
    Create cohort community bag for given motif labels, transition matrix,
    cut tree, and number of clusters. (markov chain to tree -> community detection)

    Parameters
    ----------
    config : dict
        Configuration parameters.
    motif_labels : List[np.ndarray]
        List of motif label arrays.
    trans_mat_full : np.ndarray
        Full transition matrix.
    cut_tree : int | None
        Cut line for tree.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm : str
        Which segmentation algorithm to use. Options are 'hmm' or 'kmeans'.

    Returns
    -------
    List
        List of community bags.
    """
    communities_all = []
    unique_labels, usage_full = np.unique(motif_labels, return_counts=True)
    labels_usage = dict()
    for la, u in zip(unique_labels, usage_full):
        labels_usage[str(la)] = u / np.sum(usage_full)
    T = graph_to_tree(
        motif_usage=usage_full,
        transition_matrix=trans_mat_full,
        n_clusters=n_clusters,
        merge_sel=1,
    )
    results_dir = os.path.join(
        config["project_path"],
        "results",
        "community_cohort",
        segmentation_algorithm + "-" + str(n_clusters),
    )
    nx.write_graphml(T, os.path.join(results_dir, "tree.graphml"))
    draw_tree(
        T=T,
        fig_width=n_clusters,
        usage_dict=labels_usage,
        save_to_file=True,
        show_figure=False,
        results_dir=results_dir,
    )
    # nx.write_gpickle(T, 'T.gpickle')

    if cut_tree is not None:
        # communities_all = traverse_tree_cutline(T, cutline=cut_tree)
        communities_all = bag_nodes_by_cutline(
            tree=T,
            cutline=cut_tree,
            root="Root",
        )
        logger.info("Communities bag:")
        for ci, comm in enumerate(communities_all):
            logger.info(f"Community {ci}: {comm}")
    else:
        plt.pause(0.5)
        flag_1 = "no"
        while flag_1 == "no":
            cutline = int(input("Where do you want to cut the Tree? 0/1/2/3/..."))
            # community_bag = traverse_tree_cutline(T, cutline=cutline)
            community_bag = bag_nodes_by_cutline(
                tree=T,
                cutline=cutline,
                root="Root",
            )
            logger.info(community_bag)
            flag_2 = input("\nAre all motifs in the list? (yes/no/restart)")
            if flag_2 == "no":
                while flag_2 == "no":
                    add = input("Extend list or add in the end? (ext/end)")
                    if add == "ext":
                        motif_idx = int(input("Which motif number? "))
                        list_idx = int(input("At which position in the list? (pythonic indexing starts at 0) "))
                        community_bag[list_idx].append(motif_idx)
                    if add == "end":
                        motif_idx = int(input("Which motif number? "))
                        community_bag.append([motif_idx])
                        logger.info(community_bag)
                    flag_2 = input("\nAre all motifs in the list? (yes/no/restart)")
            if flag_2 == "restart":
                continue
            if flag_2 == "yes":
                communities_all = community_bag
                flag_1 = "yes"
    return communities_all


def get_cohort_community_labels(
    motif_labels: List[np.ndarray],
    cohort_community_bag: list,
) -> List[np.ndarray]:
    """
    Transform kmeans/hmm parameterized latent vector motifs into communities.
    Get cohort community labels for given labels, and community bags.

    Parameters
    ----------
    labels : List[np.ndarray]
        List of label arrays.
    cohort_community_bag : np.ndarray
        List of community bags. Dimensions: (n_communities, n_clusters_in_community)

    Returns
    -------
    List[np.ndarray]
        List of cohort community labels for each file.
    """
    community_labels_all = []
    num_comm = len(cohort_community_bag)
    community_labels = np.zeros_like(motif_labels)
    for i in range(num_comm):
        clust = np.asarray(cohort_community_bag[i])
        for j in range(len(clust)):
            find_clust = np.where(motif_labels == clust[j])[0]
            community_labels[find_clust] = i
    community_labels_all.append(community_labels)
    return community_labels_all


def save_cohort_community_labels_per_session(
    config: dict,
    sessions: List[str],
    model_name: str,
    n_clusters: int,
    segmentation_algorithm: str,
    cohort_community_bag: list,
) -> None:
    for idx, session in enumerate(sessions):
        path_to_dir = os.path.join(
            config["project_path"],
            "results",
            session,
            model_name,
            segmentation_algorithm + "-" + str(n_clusters),
            "",
        )
        file_labels = np.load(
            os.path.join(
                path_to_dir,
                str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session + ".npy",
            )
        )
        community_labels = get_cohort_community_labels(
            motif_labels=file_labels,
            cohort_community_bag=cohort_community_bag,
        )
        if not os.path.exists(os.path.join(path_to_dir, "community")):
            os.mkdir(os.path.join(path_to_dir, "community"))
        np.save(
            os.path.join(
                path_to_dir,
                "community",
                f"cohort_community_label_{session}.npy",
            ),
            np.array(community_labels[0]),
        )


@save_state(model=CommunityFunctionSchema)
def community(
    config: dict,
    cut_tree: int = 3,
    save_logs: bool = True,
) -> None:
    """
    Perform community analysis.
    Fills in the values in the "community" key of the states.json file.
    Saves results files at:
    - project_name/
        - results/
            - community_cohort/
                - segmentation_algorithm-n_clusters/
                    - cohort_community_bag.npy
                    - cohort_community_label.npy
                    - cohort_segmentation_algorithm_label.npy
                    - cohort_transition_matrix.npy
                    - hierarchy.pkl
            - session_name/
                - model_name/
                    - segmentation_algorithm-n_clusters/
                        - community/
                            - cohort_community_label_session_name.npy

    Parameters
    ----------
    config : dict
        Configuration parameters.
    cut_tree : int, optional
        Cut line for tree. Defaults to None.
    save_logs : bool, optional
        Whether to save logs. Defaults to True.

    Returns
    -------
    None
    """
    try:
        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "community.log"
            logger_config.add_file_handler(str(log_path))

        model_name = config["model_name"]
        n_clusters = config["n_clusters"]
        segmentation_algorithms = config["segmentation_algorithms"]

        # Get sessions
        if config["all_data"] in ["Yes", "yes", "True", "true", True]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                config=config,
                action_message="run community analysis",
            )

        logger.info("---------------------------------------------------------------------")
        logger.info(f"Community analysis for model: {model_name} \n")
        for seg in segmentation_algorithms:
            logger.info(f"Community analysis for segmentation algorithm {seg} with {n_clusters} clusters")
            path_to_dir = Path(
                os.path.join(
                    config["project_path"],
                    "results",
                    "community_cohort",
                    seg + "-" + str(n_clusters),
                )
            )
            if not path_to_dir.exists():
                path_to_dir.mkdir(parents=True, exist_ok=True)

            # STEP 1
            cohort_motif_labels, cohort_motif_counts = get_motif_labels(
                config=config,
                sessions=sessions,
                model_name=model_name,
                n_clusters=n_clusters,
                segmentation_algorithm=seg,
            )
            np.save(
                os.path.join(
                    path_to_dir,
                    "cohort_" + seg + "_label" + ".npy",
                ),
                cohort_motif_labels,
            )
            logger.info(f"Cohort motif labels from {seg} saved")
            np.save(
                os.path.join(
                    path_to_dir,
                    "cohort_" + seg + "_count" + ".npy",
                ),
                cohort_motif_counts,
            )
            logger.info(f"Cohort motif counts from {seg} saved")
            logger.info(cohort_motif_counts)

            # STEP 2
            _, trans_mat_full, _ = get_adjacency_matrix(
                labels=cohort_motif_labels,
                n_clusters=n_clusters,
            )
            np.save(
                os.path.join(
                    path_to_dir,
                    "cohort_transition_matrix" + ".npy",
                ),
                trans_mat_full,
            )
            logger.info("Cohort transition matrix saved")

            # STEP 3
            cohort_community_bag = create_cohort_community_bag(
                config=config,
                motif_labels=cohort_motif_labels,
                trans_mat_full=trans_mat_full,
                cut_tree=cut_tree,
                n_clusters=n_clusters,
                segmentation_algorithm=seg,
            )
            # convert cohort_community_bag to dtype object numpy array because it is an inhomogeneous list
            cohort_community_bag = np.array(cohort_community_bag, dtype=object)
            np.save(
                os.path.join(
                    path_to_dir,
                    "cohort_community_bag" + ".npy",
                ),
                cohort_community_bag,
            )
            logger.info("Community bag saved")

            # STEP 4
            community_labels_all = get_cohort_community_labels(
                motif_labels=cohort_motif_labels,
                cohort_community_bag=cohort_community_bag,
            )
            np.save(
                os.path.join(
                    path_to_dir,
                    "cohort_community_label" + ".npy",
                ),
                community_labels_all,
            )
            logger.info("Community labels saved")

            with open(os.path.join(path_to_dir, "hierarchy" + ".pkl"), "wb") as fp:  # Pickling
                pickle.dump(cohort_community_bag, fp)

            # Added by Luiz - 11/10/2024
            # Saves the full community labels list for each one of sessions
            # This is useful for further analysis when cohort=True
            save_cohort_community_labels_per_session(
                config=config,
                sessions=sessions,
                model_name=model_name,
                n_clusters=n_clusters,
                segmentation_algorithm=seg,
                cohort_community_bag=cohort_community_bag,
            )

    except Exception as e:
        logger.exception(f"Error in community_analysis: {e}")
        raise e
    finally:
        logger_config.remove_file_handler()
