from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from vame.io.load_poses import read_pose_estimation_file


def pose_estimation_inspection(
    config,
    read_from_variable: str = "position_raw",
    save_to_file: bool = False,
    show_figure: bool = True,
) -> None:
    """
    Inspect pose estimation data for quality and completeness.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    read_from_variable : str
        Name of the variable to read the raw position data from.

    Returns
    -------
    None
    """
    project_path = config["project_path"]
    sessions = config["session_names"]
    pose_confidence = config["pose_confidence"]
    all_confidence_values = None
    for session in sessions:
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        confidence = ds["confidence"]  # shape: (time, keypoints, individuals)
        individuals = ds["individuals"].values
        keypoints = ds["keypoints"].values

        if all_confidence_values is None:
            all_confidence_values = confidence.values.copy()
        else:
            all_confidence_values = np.concatenate(
                (all_confidence_values, confidence.values),
                axis=0,
            )

    for i in range(len(individuals)):
        for k in range(len(keypoints)):
            perc = (all_confidence_values[:, k, i] < pose_confidence).mean() * 100
            print(f"{individuals[i]} | {keypoints[k]} - samples below confidence reference: {perc:.1f}%")

    # Plot pose estimation inspection results
    if all_confidence_values is not None:
        plot_pose_estimation_inspection_matplotlib(
            config=config,
            confidence_data=all_confidence_values,
            keypoint_names=keypoints,
            save_to_file=save_to_file,
            show_figure=show_figure,
        )


def plot_pose_estimation_inspection_matplotlib(
    config,
    confidence_data: np.ndarray,
    keypoint_names: np.ndarray,
    save_to_file: bool = False,
    show_figure: bool = True,
) -> None:
    """
    Plot pose estimation inspection results using matplotlib with multiple subplots.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    confidence_data : np.ndarray
        Confidence data array with shape (time, keypoints, individuals).
    keypoint_names : np.ndarray
        Array of keypoint names.

    Returns
    -------
    None
    """
    project_path = config["project_path"]
    confidence_reference = config["pose_confidence"]
    n_keypoints = len(keypoint_names)

    # Calculate subplot layout: 2 columns, variable rows
    n_cols = 2
    n_rows = (n_keypoints + n_cols - 1) // n_cols  # Ceiling division

    # Calculate figure height based on number of rows
    fig_height = max(6, n_rows * 3)  # Minimum 6 inches, 3 inches per row

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, fig_height))
    fig.suptitle("Confidence Score Distributions", fontsize=16, y=0.98)

    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i, keypoint in enumerate(keypoint_names):
        ax = axes_flat[i]

        # Extract confidence values for this keypoint (first individual)
        confidence_array = confidence_data[:, i, 0]

        # Calculate percentage below reference
        below_reference = np.sum(confidence_array < confidence_reference)
        total_samples = len(confidence_array)
        percentage_below = (below_reference / total_samples) * 100

        # Create histogram with percentage normalization
        total_samples = len(confidence_array)
        weights = np.full(total_samples, 100.0 / total_samples)

        ax.hist(
            confidence_array,
            bins=50,
            weights=weights,
            color="#8d93b5",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.7
        )

        # Add reference line
        ax.axvline(
            confidence_reference,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Reference = {confidence_reference:.2f}"
        )

        # Set title with keypoint name and percentage
        ax.set_title(
            f"{keypoint}\n{percentage_below:.1f}% below reference",
            fontsize=12,
            pad=10
        )

        # Set labels and formatting
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Percentage (%)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(n_keypoints, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle

    if save_to_file:
        save_fig_path = Path(project_path) / "reports" / "figures" / "pose_estimation_confidence_distribution.png"
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_fig_path))

    if show_figure:
        plt.show()
    else:
        plt.close(fig)
