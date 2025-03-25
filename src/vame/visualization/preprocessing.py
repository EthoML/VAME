from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

from vame.io.load_poses import read_pose_estimation_file


def visualize_preprocessing_scatter(
    config: dict,
    session_index: int = 0,
    frames: list = [],
    original_positions_key: str = "position",
    cleaned_positions_key: str = "position_cleaned_lowconf",
    aligned_positions_key: str = "position_egocentric_aligned",
    save_to_file: bool = False,
    show_figure: bool = True,
):
    """
    Visualize the preprocessing results by plotting the original, cleaned low-confidence,
    and egocentric aligned positions of the keypoints in a scatter plot.
    """
    project_path = config["project_path"]
    sessions = config["session_names"]
    session = sessions[session_index]

    # Read session data
    file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
    _, _, ds = read_pose_estimation_file(file_path=file_path)

    original_positions = ds[original_positions_key].values
    cleaned_positions = ds[cleaned_positions_key].values
    aligned_positions = ds[aligned_positions_key].values
    keypoints_labels = ds.keypoints.values

    if not frames:
        frames = [int(i * len(original_positions)) for i in [0.1, 0.3, 0.5, 0.7, 0.9]]
    num_frames = len(frames)

    fig, axes = plt.subplots(num_frames, 3, figsize=(18, 6 * num_frames))  # Adjusted figure width for better spacing

    for i, frame in enumerate(frames):
        # Compute dynamic limits for the original positions
        x_orig = original_positions[frame, 0, :, 0]
        y_orig = original_positions[frame, 1, :, 0]

        # Identify keypoints that are NaN
        nan_keypoints = [
            keypoints_labels[k] for k in range(len(keypoints_labels)) if np.isnan(x_orig[k]) or np.isnan(y_orig[k])
        ]

        # Check if original positions contain all NaNs
        if np.all(np.isnan(x_orig)) or np.all(np.isnan(y_orig)):
            ax_original = axes[i, 0]
            ax_original.set_title(f"Original - Frame {frame} (All NaNs)", fontsize=14, color="red")
            ax_original.axis("off")  # Hide axis since there is no data to plot
        else:
            x_min, x_max = np.nanmin(x_orig) - 10, np.nanmax(x_orig) + 10  # Add a margin
            y_min, y_max = np.nanmin(y_orig) - 10, np.nanmax(y_orig) + 10

            ax_original = axes[i, 0]
            ax_original.scatter(x_orig, y_orig, c="blue", label="Original")
            for k, (x, y) in enumerate(zip(x_orig, y_orig)):
                ax_original.text(x, y, keypoints_labels[k], fontsize=10, color="blue")

            # Include NaN keypoints in the title
            if nan_keypoints:
                nan_text = ", ".join(nan_keypoints)
                title_text = f"Original - Frame {frame}\nNaNs: {nan_text}"
            else:
                title_text = f"Original - Frame {frame}"

            ax_original.set_title(title_text, fontsize=14)

            ax_original.set_xlabel("X", fontsize=12)
            ax_original.set_ylabel("Y", fontsize=12)
            ax_original.axhline(0, color="gray", linestyle="--")
            ax_original.axvline(0, color="gray", linestyle="--")
            # Ensure square aspect by making the limits have equal range
            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range, y_range)
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            ax_original.set_xlim(x_center - max_range/2, x_center + max_range/2)
            ax_original.set_ylim(y_center - max_range/2, y_center + max_range/2)
            ax_original.set_aspect('equal')

        # Compute dynamic limits for the cleaned positions
        x_cleaned = cleaned_positions[frame, 0, :, 0]
        y_cleaned = cleaned_positions[frame, 1, :, 0]
        x_min_cleaned, x_max_cleaned = x_cleaned.min() - 10, x_cleaned.max() + 10  # Add a margin
        y_min_cleaned, y_max_cleaned = y_cleaned.min() - 10, y_cleaned.max() + 10

        # Centralized Cleaned positions
        ax_cleaned = axes[i, 1]
        ax_cleaned.scatter(x_cleaned, y_cleaned, c="orange", label="Cleaned Low-Conf")
        for k, (x, y) in enumerate(zip(x_cleaned, y_cleaned)):
            ax_cleaned.text(x, y, keypoints_labels[k], fontsize=10, color="orange")
        ax_cleaned.set_title(f"Cleaned - Frame {frame}", fontsize=14)
        ax_cleaned.set_xlabel("X", fontsize=12)
        ax_cleaned.set_ylabel("Y", fontsize=12)
        ax_cleaned.axhline(0, color="gray", linestyle="--")
        ax_cleaned.axvline(0, color="gray", linestyle="--")
        # Ensure square aspect by making the limits have equal range
        x_range_cleaned = x_max_cleaned - x_min_cleaned
        y_range_cleaned = y_max_cleaned - y_min_cleaned
        max_range_cleaned = max(x_range_cleaned, y_range_cleaned)
        x_center_cleaned = (x_max_cleaned + x_min_cleaned) / 2
        y_center_cleaned = (y_max_cleaned + y_min_cleaned) / 2
        ax_cleaned.set_xlim(x_center_cleaned - max_range_cleaned/2, x_center_cleaned + max_range_cleaned/2)
        ax_cleaned.set_ylim(y_center_cleaned - max_range_cleaned/2, y_center_cleaned + max_range_cleaned/2)
        ax_cleaned.set_aspect('equal')

        # Compute dynamic limits for the aligned positions
        x_aligned = aligned_positions[frame, 0, :, 0]
        y_aligned = aligned_positions[frame, 1, :, 0]
        x_min_aligned, x_max_aligned = x_aligned.min() - 10, x_aligned.max() + 10  # Add a margin
        y_min_aligned, y_max_aligned = y_aligned.min() - 10, y_aligned.max() + 10

        # Centralized Aligned positions
        ax_aligned = axes[i, 2]
        ax_aligned.scatter(x_aligned, y_aligned, c="green", label="Egocentric Aligned")
        for k, (x, y) in enumerate(zip(x_aligned, y_aligned)):
            ax_aligned.text(x, y, keypoints_labels[k], fontsize=10, color="green")
        ax_aligned.set_title(f"Aligned - Frame {frame}", fontsize=14)
        ax_aligned.set_xlabel("X", fontsize=12)
        ax_aligned.set_ylabel("Y", fontsize=12)
        ax_aligned.axhline(0, color="gray", linestyle="--")
        ax_aligned.axvline(0, color="gray", linestyle="--")
        # Ensure square aspect by making the limits have equal range
        x_range_aligned = x_max_aligned - x_min_aligned
        y_range_aligned = y_max_aligned - y_min_aligned
        max_range_aligned = max(x_range_aligned, y_range_aligned)
        x_center_aligned = (x_max_aligned + x_min_aligned) / 2
        y_center_aligned = (y_max_aligned + y_min_aligned) / 2
        ax_aligned.set_xlim(x_center_aligned - max_range_aligned/2, x_center_aligned + max_range_aligned/2)
        ax_aligned.set_ylim(y_center_aligned - max_range_aligned/2, y_center_aligned + max_range_aligned/2)
        ax_aligned.set_aspect('equal')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, top=0.9)  # Control spacing and add top margin for title
    plt.tight_layout(pad=1.5)  # Reduced padding for tighter layout

    # Add a figure-level title after layout adjustments
    fig.suptitle(
        f"{session}, Confidence threshold: {config['pose_confidence']}",
        fontsize=16,
        y=1.01,  # Position the title higher
    )

    if save_to_file:
        save_fig_path = Path(project_path) / "reports" / "figures" / f"{session}_preprocessing_scatter.png"
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_fig_path))

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def visualize_preprocessing_timeseries(
    config: dict,
    session_index: int = 0,
    n_samples: int = 1000,
    original_positions_key: str = "position",
    aligned_positions_key: str = "position_egocentric_aligned",
    processed_positions_key: str = "position_processed",
    save_to_file: bool = False,
    show_figure: bool = True,
):
    """
    Visualize the preprocessing results by plotting the original, aligned, and processed positions
    of the keypoints in a timeseries plot.
    """
    project_path = config["project_path"]
    sessions = config["session_names"]
    session = sessions[session_index]

    # Read session data
    file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
    _, _, ds = read_pose_estimation_file(file_path=file_path)

    fig, ax = plt.subplots(6, 1, figsize=(10, 16))  # Adjusted for 6 subplots

    individual = "individual_0"
    keypoints_labels = ds.keypoints.values

    # Create a colormap with distinguishable colors
    cmap = get_cmap("tab10") if len(keypoints_labels) <= 10 else get_cmap("tab20")
    colors = [cmap(i / len(keypoints_labels)) for i in range(len(keypoints_labels))]

    for i, kp in enumerate(keypoints_labels):
        sel_x = dict(
            individuals=individual,
            keypoints=kp,
            space="x",
        )
        sel_y = dict(
            individuals=individual,
            keypoints=kp,
            space="y",
        )

        # Original positions (first two subplots)
        ds[original_positions_key].sel(**sel_x)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[0],
            label=kp,
            color=colors[i],
        )
        ds[original_positions_key].sel(**sel_y)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[1],
            label=kp,
            color=colors[i],
        )

        # Aligned positions (next two subplots)
        ds[aligned_positions_key].sel(**sel_x)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[2],
            label=kp,
            color=colors[i],
        )
        ds[aligned_positions_key].sel(**sel_y)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[3],
            label=kp,
            color=colors[i],
        )

        # Processed positions (last two subplots)
        ds[processed_positions_key].sel(**sel_x)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[4],
            label=kp,
            color=colors[i],
        )
        ds[processed_positions_key].sel(**sel_y)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[5],
            label=kp,
            color=colors[i],
        )

    # Set common labels for Y axes
    ax[0].set_ylabel(
        "Original Allocentric X",
        fontsize=12,
    )
    ax[1].set_ylabel(
        "Original Allocentric Y",
        fontsize=12,
    )
    ax[2].set_ylabel(
        "Aligned Egocentric X",
        fontsize=12,
    )
    ax[3].set_ylabel(
        "Aligned Egocentric Y",
        fontsize=12,
    )
    ax[4].set_ylabel(
        "Processed Egocentric X",
        fontsize=12,
    )
    ax[5].set_ylabel(
        "Processed Egocentric Y",
        fontsize=12,
    )

    # Labels for X axes
    for idx, a in enumerate(ax):
        a.set_title("")
        if idx % 2 == 0:
            a.set_xlabel("")
        else:
            a.set_xlabel(
                "Time",
                fontsize=10,
            )

    # Adjust padding
    fig.subplots_adjust(hspace=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.2)

    # Add a single legend for all subplots
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.98),
    )

    if save_to_file:
        save_fig_path = Path(project_path) / "reports" / "figures" / f"{session}_preprocessing_timeseries.png"
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            str(save_fig_path),
        )

    if show_figure:
        plt.show()
    else:
        plt.close(fig)
