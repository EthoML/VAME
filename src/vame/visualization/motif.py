from pathlib import Path
from typing import Literal, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_motif_thresholding_v0(
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
    plt.axhline(y=threshold, color='black', linestyle='--', label=f"{threshold}% threshold")  # threshold line
    plt.xlabel("Sorted Session Index")
    plt.ylabel("Motif Percentage (%)")
    plt.title("Motif Count as Percentage")
    plt.legend()
    plt.show()


def plot_motif_thresholding(
    config: dict,
    segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
    n_clusters: int | None = None,
    threshold: float = 1.0,
    show_motif_ids: bool = True,
    alpha: float = 0.3,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot motif usage profile in comparison to threshold.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing project settings.
    segmentation_algorithm : Literal["hmm", "kmeans"], optional
        Segmentation algorithm used. Defaults to "hmm".
    n_clusters : int, optional
        Number of clusters/motifs. Uses config["n_clusters"] if None.
    threshold : float, optional
        Percentage threshold for motif significance. Defaults to 1.0.
    show_motif_ids : bool, optional
        If True, show motif IDs on x-axis. Defaults to True.
    alpha : float, optional
        Transparency for individual session lines. Defaults to 0.3.
    figsize : Tuple[int, int], optional
        Figure size (width, height). Defaults to (12, 6).
    """
    results_path = Path(config["project_path"]) / 'results'

    if n_clusters is None:
        n_clusters = config["n_clusters"]

    # Collect motif usage data for all sessions
    session_usage_data = []
    session_names = []

    for session in config["session_names"]:
        motif_usage_file = Path(results_path) / session / config["model_name"] / f"{segmentation_algorithm}-{n_clusters}" / f"motif_usage_{session}.npy"

        if not motif_usage_file.exists():
            print(f"Warning: Motif usage file not found for session {session}")
            continue

        session_motif_count = np.load(motif_usage_file)

        # Validate cluster count consistency
        if len(session_motif_count) != n_clusters:
            print(f"Warning: Session {session} has {len(session_motif_count)} motifs, expected {n_clusters}")
            continue

        # Convert to percentages
        total_motifs = np.sum(session_motif_count)
        if total_motifs == 0:
            print(f"Warning: Session {session} has no motifs")
            continue

        session_motif_percentages = (session_motif_count / total_motifs) * 100
        session_usage_data.append(session_motif_percentages)
        session_names.append(session)

    if not session_usage_data:
        raise ValueError("No valid session data found")

    # Shape (n_sessions, n_motifs)
    session_usage_data = np.array(session_usage_data)

    # Calculate mean usage per motif across sessions
    mean_usage_per_motif = np.mean(session_usage_data, axis=0)

    # Always sort by mean usage but keep track of original motif IDs
    motif_ids = np.arange(n_clusters)
    sort_indices = np.argsort(mean_usage_per_motif)[::-1]  # descending order
    motif_order = motif_ids[sort_indices]
    display_usage_data = session_usage_data[:, sort_indices]
    display_mean_usage = mean_usage_per_motif[sort_indices]

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot individual session lines
    for i, session in enumerate(session_names):
        plt.plot(display_usage_data[i], color='blue', alpha=alpha, linewidth=0.8)

    # Plot mean line
    plt.plot(display_mean_usage, color='red', linewidth=2, label="Mean motif usage", zorder=3)

    # Add threshold line
    plt.axhline(
        y=threshold,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=f"{threshold}% threshold",
    )

    # Identify motifs above threshold
    motifs_above_threshold = motif_order[display_mean_usage >= threshold]

    # Add two-tier marker system for all motifs
    above_threshold_mask = display_mean_usage >= threshold
    below_threshold_mask = ~above_threshold_mask

    # Large markers for motifs above threshold
    if np.any(above_threshold_mask):
        plt.scatter(
            np.where(above_threshold_mask)[0],
            display_mean_usage[above_threshold_mask],
            color='red',
            s=50,
            zorder=5,
            alpha=0.8,
            label=f"Motifs above {threshold}%",
        )

    # Small markers for motifs below threshold
    if np.any(below_threshold_mask):
        plt.scatter(
            np.where(below_threshold_mask)[0],
            display_mean_usage[below_threshold_mask],
            color='gray',
            s=25,
            zorder=6,
            alpha=0.6,
            label=f"Motifs below {threshold}%"
        )

    # Customize plot
    plt.xlabel("Motif Rank (by mean usage)")
    plt.ylabel("Motif Usage (%)")

    title = f"Motif Usage Analysis ({segmentation_algorithm.upper()}, {n_clusters} clusters) - Sorted by Mean Usage"
    plt.title(title)

    # Set x-axis labels
    if show_motif_ids:
        plt.xticks(range(len(motif_order)), motif_order)

    # Add legend
    # Create a single entry for session lines
    plt.plot([], [], color='blue', alpha=alpha, linewidth=0.8,
             label=f'Individual sessions (n={len(session_names)})')
    plt.legend()

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Print summary information
    print(f"\nMotif Usage Analysis Summary:")
    print(f"- Total sessions analyzed: {len(session_names)}")
    print(f"- Number of motifs: {n_clusters}")
    print(f"- Threshold: {threshold}%")
    print(f"- Motifs above threshold: {len(motifs_above_threshold)}")

    if len(motifs_above_threshold) > 0:
        for motif_id in motifs_above_threshold:
            original_idx = np.where(motif_order == motif_id)[0][0]
            usage = display_mean_usage[original_idx]
            print(f"  Motif {motif_id}: {usage:.2f}%")
    else:
        print("- No motifs above threshold")
