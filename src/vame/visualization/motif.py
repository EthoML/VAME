from pathlib import Path
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt


def visualize_motif_thresholding(
    config: dict,
    segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
    n_clusters: int | None = None,
    threshold: float = 1.0,
    show_figure: bool = True,
    save_to_file: bool = False,
    fig_size: tuple = (10, 6),
) -> None:
    """
    Visualizes the motif usage with thresholding.

    Parameters
    ----------
    segmentation_algorithm : Literal["hmm", "kmeans"], optional
        Segmentation algorithm, by default "hmm".
    n_clusters : Optional[int], optional
        Number of clusters, by default None. If None, it uses the value from config["n_clusters"].
    threshold : float, optional
        Threshold, by default 1.0.
    show_figure : bool, optional
        Whether to show the figure, by default True.
    save_to_file : bool, optional
        Whether to save the figure to file, by default False.
    fig_size : tuple, optional
        Figure size, by default (10, 6).

    Returns
    -------
    None
    """
    results_path = Path(config["project_path"]) / 'results'

    if n_clusters is None:
        n_clusters = config["n_clusters"]

    fig = plt.figure(figsize=fig_size)

    all_session_m_counts = []
    for s in config["session_names"]:
        motif_usage_file = Path(results_path) / s / config["model_name"] / f"{segmentation_algorithm}-{n_clusters}" / f"motif_usage_{s}.npy"
        session_motif_count = np.load(motif_usage_file)
        session_motif_count_desc = np.sort(session_motif_count)[::-1]  # sort by descending order
        total_motifs = np.sum(session_motif_count_desc)
        sess_motif_count_desc_perc = (session_motif_count_desc / total_motifs) * 100
        all_session_m_counts.append(sess_motif_count_desc_perc)
        plt.plot(sess_motif_count_desc_perc, color='blue', linewidth=0.5)

    all_session_m_counts = np.array(all_session_m_counts)
    mean_session_m_counts = np.mean(all_session_m_counts, axis=0)

    plt.plot([], [], color='blue', label='Session sorted motif')  # single blue line key
    plt.plot(mean_session_m_counts, color='r', label="Sorted motif mean")  # red mean line
    plt.axhline(y=threshold, color='black', linestyle='--', label=f"{threshold}% usage threshold")  # threshold line
    plt.xlabel("Sorted Motif Index")
    plt.ylabel("Motif Usage in Percentage (%)")
    plt.title("Sorted Motif Usage")
    plt.legend()

    if save_to_file:
        save_path = Path(config["project_path"]) / "reports" / "figures"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"motif_thresholding_{segmentation_algorithm}_{n_clusters}.png")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)
