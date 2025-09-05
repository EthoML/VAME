from pathlib import Path
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt


def plot_motif_thresholding(
    config: dict,
    segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
    n_clusters: int | None = None,
    threshold: float = 1.0
) -> None:
    results_path = Path(config["project_path"]) / 'results'

    if n_clusters is None:
        n_clusters = config["n_clusters"]

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
    plt.show()
