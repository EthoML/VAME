from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from vame.util.data_manipulation import read_pose_estimation_file


def visualize_preprocessing_scatter(
    config: dict,
    session_index: int = 0,
    frames: list = [],
    save_to_file: bool = False,
):
    """
    Visualize the preprocessing results by plotting the original and aligned positions
    of the keypoints in a scatter plot.
    """
    project_path = config["project_path"]
    sessions = config["session_names"]
    session = sessions[session_index]

    # Read session data
    file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
    _, _, ds = read_pose_estimation_file(file_path=file_path)

    original_positions = ds["position"].values
    processed_positions = ds["position_processed"].values
    keypoints_labels = ds.keypoints.values

    if not frames:
        frames = [int(i * len(original_positions)) for i in [0.1, 0.3, 0.5, 0.7, 0.9]]
    num_frames = len(frames)

    fig, axes = plt.subplots(num_frames, 2, figsize=(14, 6 * num_frames))  # Increased figure size

    for i, frame in enumerate(frames):
        # Compute dynamic limits for the original positions
        x_orig, y_orig = original_positions[frame, 0, :, 0], original_positions[frame, 0, :, 1]
        x_orig -= x_orig[0]  # Centralize around the first keypoint
        y_orig -= y_orig[0]
        x_min, x_max = x_orig.min() - 10, x_orig.max() + 10  # Add a margin
        y_min, y_max = y_orig.min() - 10, y_orig.max() + 10

        # Centralized Original positions
        ax_original = axes[i, 0]
        ax_original.scatter(x_orig, y_orig, c="blue", label="Original")
        for k, (x, y) in enumerate(zip(x_orig, y_orig)):
            ax_original.text(x, y, keypoints_labels[k], fontsize=8, color="blue")
        ax_original.set_title(f"Original - Frame {frame}", fontsize=10)
        ax_original.set_xlabel("X", fontsize=8)
        ax_original.set_ylabel("Y", fontsize=8)
        ax_original.axhline(0, color="gray", linestyle="--")
        ax_original.axvline(0, color="gray", linestyle="--")
        ax_original.axis("equal")
        ax_original.set_xlim(x_min, x_max)
        ax_original.set_ylim(y_min, y_max)

        # Compute dynamic limits for the aligned positions
        x_aligned, y_aligned = processed_positions[frame, 0, :, 0], processed_positions[frame, 0, :, 1]
        x_min_aligned, x_max_aligned = x_aligned.min() - 10, x_aligned.max() + 10  # Add a margin
        y_min_aligned, y_max_aligned = y_aligned.min() - 10, y_aligned.max() + 10

        # Centralized Aligned positions
        ax_aligned = axes[i, 1]
        ax_aligned.scatter(x_aligned, y_aligned, c="green", label="Aligned")
        for k, (x, y) in enumerate(zip(x_aligned, y_aligned)):
            ax_aligned.text(x, y, keypoints_labels[k], fontsize=8, color="green")
        ax_aligned.set_title(f"Aligned - Frame {frame}", fontsize=10)
        ax_aligned.set_xlabel("X", fontsize=8)
        ax_aligned.set_ylabel("Y", fontsize=8)
        ax_aligned.axhline(0, color="gray", linestyle="--")
        ax_aligned.axvline(0, color="gray", linestyle="--")
        ax_aligned.axis("equal")
        ax_aligned.set_xlim(x_min_aligned, x_max_aligned)
        ax_aligned.set_ylim(y_min_aligned, y_max_aligned)

    plt.tight_layout(pad=2.0)  # Add padding to reduce overlap

    if save_to_file:
        save_fig_path = Path(project_path) / "reports" / "figures" / f"{session}_preprocessing_scatter.png"
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_fig_path))


def visualize_preprocessing_timeseries(
    config: dict,
    session_index: int = 0,
    n_samples: int = 1000,
    save_to_file: bool = False,
):
    """
    Visualize the preprocessing results by plotting the original and aligned positions
    of the keypoints in a timeseries plot.
    """
    project_path = config["project_path"]
    sessions = config["session_names"]
    session = sessions[session_index]

    # Read session data
    file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
    _, _, ds = read_pose_estimation_file(file_path=file_path)

    fig, ax = plt.subplots(4, 1, figsize=(10, 12))

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

        ds.position.sel(**sel_x)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[0],
            label=kp,
            color=colors[i],
        )
        ds.position.sel(**sel_y)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[1],
            label=kp,
            color=colors[i],
        )
        ds.position_processed.sel(**sel_x)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[2],
            label=kp,
            color=colors[i],
        )
        ds.position_processed.sel(**sel_y)[0:n_samples].plot(
            linewidth=1.5,
            ax=ax[3],
            label=kp,
            color=colors[i],
        )

    ax[0].set_title("")
    ax[1].set_title("")
    ax[2].set_title("")
    ax[3].set_title("")

    ax[0].set_xlabel("")
    ax[1].set_xlabel("")
    ax[2].set_xlabel("")

    ax[0].set_ylabel("Allocentric X")
    ax[1].set_ylabel("Allocentric Y")
    ax[2].set_ylabel("Egocentric X")
    ax[3].set_ylabel("Egocentric Y")

    # Add a single legend for all subplots
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_to_file:
        save_fig_path = Path(project_path) / "reports" / "figures" / f"{session}_preprocessing_timeseries.png"
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_fig_path))
