import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

from vame.util.auxiliary import read_config
from vame.schemas.project import Parametrizations


def report(
    config: str,
    parametrization: str = "hmm",
) -> None:
    """
    Report for a project.
    """
    config_file = Path(config).resolve()
    cfg = read_config(str(config_file))
    project_path = Path(cfg["project_path"])
    n_clusters = cfg["n_cluster"]

    with open(project_path / "states" / "states.json") as f:
        project_states = json.load(f)

    pose_estimation_files = list(
        (project_path / "videos/pose_estimation").glob("*.csv")
    )
    video_files = list((project_path / "videos").glob("*.mp4"))

    # Motifs and Communities
    if (
        project_states.get("segment_session", {}).get("execution_state", "")
        != "success"
    ):
        raise Exception("Segmentation failed. Skipping motifs and communities report.")
    if project_states.get("community", {}).get("execution_state", "") != "success":
        raise Exception(
            "Community detection failed. Skipping motifs and communities report."
        )

    ml = np.load(
        project_path
        / f"results/community_cohort/{parametrization}-{n_clusters}/cohort_{parametrization}_label.npy",
        allow_pickle=True,
    )
    motif_labels = dict()
    u, c = np.unique(ml, return_counts=True)
    for uu, cc in zip(u, c):
        motif_labels[uu] = cc

    cl = np.load(
        project_path
        / f"results/community_cohort/{parametrization}-{n_clusters}/cohort_community_label.npy",
        allow_pickle=True,
    )
    community_labels = dict()
    u, c = np.unique(cl, return_counts=True)
    for uu, cc in zip(u, c):
        community_labels[uu] = cc

    community_bag = np.load(
        project_path
        / f"results/community_cohort/{parametrization}-{n_clusters}/cohort_community_bag.npy",
        allow_pickle=True,
    )

    print("Cohort communities:")
    for ii, bag in enumerate(community_bag):
        print(f"Community {ii}: {community_labels[ii]} counts")
        for jj in bag:
            print(f"    Motif {jj}: {motif_labels[jj]} counts")

    plot_community_motifs(
        motif_labels,
        community_labels,
        community_bag,
        title="Community and Motif Counts - Cohort",
    )

    # Per session file
    for f in pose_estimation_files:
        file_name = str(f.resolve()).split("/")[-1].split(".")[0]

        fml = np.load(
            project_path
            / f"results/{file_name}/VAME/{parametrization}-{n_clusters}/{n_clusters}_{parametrization}_label_{file_name}.npy",
            allow_pickle=True,
        )
        file_motif_labels = dict()
        u, c = np.unique(fml, return_counts=True)
        for uu, cc in zip(u, c):
            file_motif_labels[uu] = cc

        fcl = np.load(
            project_path
            / f"results/{file_name}/VAME/{parametrization}-{n_clusters}/community/cohort_community_label_{file_name}.npy",
            allow_pickle=True,
        )
        file_community_labels = dict()
        u, c = np.unique(fcl, return_counts=True)
        for uu, cc in zip(u, c):
            file_community_labels[uu] = cc

        plot_community_motifs(
            file_motif_labels,
            file_community_labels,
            community_bag,
            title=f"Community and Motif Counts - {file_name}",
        )


def plot_community_motifs(
    motif_labels,
    community_labels,
    community_bag,
    title: str = "Community and Motif Counts",
):
    """
    Generates a bar plot to represent community and motif counts with percentages.
    """
    # Extract and sort community indices and counts
    communities = list(community_labels.items())  # list of (community, count)
    # Sort communities by count in decreasing order
    communities.sort(key=lambda x: x[1], reverse=True)
    community_indices = [community for community, count in communities]
    community_counts = [count for community, count in communities]
    total_community_counts = sum(community_counts)
    community_percentages = [
        (count / total_community_counts) * 100 for count in community_counts
    ]

    # Define positions and bar widths
    bar_width = 0.8
    motif_bar_width = bar_width * 0.9  # Use 90% of community bar width for motifs

    # Create the bar plot with a secondary y-axis
    fig, ax1 = plt.subplots(figsize=(12, 7))
    bar_positions = np.arange(len(community_indices))

    # Plot community bars (background)
    community_bars = ax1.bar(
        bar_positions,
        community_counts,
        width=bar_width,
        label="Community Counts",
        alpha=0.6,
        color="gray",
    )

    # Add percentage values on top of community bars
    for bar, percentage in zip(community_bars, community_percentages):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(community_counts) * 0.01,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    # Create a second y-axis for percentages
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels(
        [f"{(tick / total_community_counts) * 100:.1f}%" for tick in ax1.get_yticks()]
    )
    ax2.set_ylabel("Percentage")

    # Overlay motif bars within each community
    for idx, community in enumerate(community_indices):
        motifs = community_bag[community]
        # Get counts and sort motifs by count in decreasing order
        motif_counts = [(motif, motif_labels[motif]) for motif in motifs]
        motif_counts.sort(key=lambda x: x[1], reverse=True)
        motifs_sorted = [motif for motif, count in motif_counts]
        counts_sorted = [count for motif, count in motif_counts]
        total_motif_counts = sum(counts_sorted)
        motif_percentages = [
            (count / total_motif_counts) * 100 for count in counts_sorted
        ]

        num_motifs = len(motifs_sorted)
        # Adjust motif bar width to fill the community bar width
        if num_motifs > 0:
            motif_width = (
                motif_bar_width / num_motifs * 0.9
            )  # Slightly reduce width to create space between bars
        else:
            motif_width = motif_bar_width

        # Center motif bars within the community bar
        total_motif_width = motif_width * num_motifs
        start_position = bar_positions[idx] - total_motif_width / 2 + motif_width / 2

        motif_positions = [start_position + i * motif_width for i in range(num_motifs)]

        # Plot the motif bars
        bars = ax1.bar(
            motif_positions,
            counts_sorted,
            width=motif_width
            * 0.9,  # Slightly reduce width to create space between bars
            label=f"Motifs in Community {community}",
        )

        # Add motif IDs as text inside the lower part of each bar
        for bar, motif_id in zip(bars, motifs_sorted):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                0,
                str(motif_id),
                ha="center",
                va="bottom",
                fontsize=9,
                color=(
                    "white" if bar.get_facecolor()[0] < 0.5 else "black"
                ),  # Contrast with bar color
            )

        # Add percentage values on top of motif bars
        for bar, percentage in zip(bars, motif_percentages):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - max(community_counts) * 0.05,
                f"{percentage:.0f}\n%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=(
                    "white" if bar.get_facecolor()[0] < 0.5 else "black"
                ),  # Contrast with bar color
            )

    # Formatting
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels([f"Community {i}" for i in community_indices])
    ax1.set_ylabel("Counts")
    ax1.set_title(title)
    ax1.legend()
    plt.tight_layout()
    plt.show()
