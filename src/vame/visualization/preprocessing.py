from pathlib import Path
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import numpy as np

from vame.io.load_poses import read_pose_estimation_file
from vame.schemas.states import save_state, PreprocessingVisualizationFunctionSchema
from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger


@save_state(model=PreprocessingVisualizationFunctionSchema)
def preprocessing_visualization(
    config: dict,
    save_to_file: bool = False,
    show_figure: bool = True,
) -> None:
    for session_index in range(len(config["session_names"])):
        visualize_preprocessing_scatter(
            config=config,
            session_index=session_index,
            save_to_file=save_to_file,
            show_figure=show_figure,
        )
        visualize_preprocessing_timeseries(
            config=config,
            session_index=session_index,
            save_to_file=save_to_file,
            show_figure=show_figure,
        )
        visualize_preprocessing_cloud(
            config=config,
            session_index=session_index,
            save_to_file=save_to_file,
            show_figure=show_figure,
        )


def visualize_preprocessing_scatter(
    config: dict,
    session_index: int = 0,
    frames: list = [],
    original_positions_key: str | None = "position",
    cleaned_positions_key: str | None = "position_cleaned_lowconf",
    aligned_positions_key: str | None = "position_egocentric_aligned",
    filtered_positions_key: str | None = "position_processed",
    scaled_positions_key: str | None = "position_scaled",
    save_to_file: bool = False,
    show_figure: bool = True,
):
    """
    Visualize the preprocessing results by plotting the positions of the keypoints in a scatter plot.
    Each position key parameter can be a string (to include that column) or None (to skip that column).

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    session_index : int, optional
        Index of the session to visualize.
    frames : list, optional
        List of frames to visualize.
    original_positions_key : str, optional
        Key for the original positions.
    cleaned_positions_key : str, optional
        Key for the low confidence cleaned positions.
    aligned_positions_key : str, optional
        Key for the egocentric aligned positions.
    filtered_positions_key : str, optional
        Key for the filtered positions.
    scaled_positions_key : str, optional
        Key for the scaled positions.
    save_to_file : bool, optional
        Whether to save the figure to a file.
    show_figure : bool, optional
        Whether to show the figure.

    Returns
    -------
    None
    """
    if not show_figure:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_path = config["project_path"]
    sessions = config["session_names"]
    session = sessions[session_index]

    # Read session data
    file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
    _, _, ds = read_pose_estimation_file(file_path=file_path)

    # Create a list of position keys and labels, filtering out None values
    position_keys = []
    position_labels = []

    if original_positions_key is not None:
        if original_positions_key not in ds.keys():
            raise KeyError(f"Key '{original_positions_key}' not found in dataset.")
        position_keys.append(original_positions_key)
        position_labels.append("Original")

    if cleaned_positions_key is not None:
        if cleaned_positions_key not in ds.keys():
            raise KeyError(f"Key '{cleaned_positions_key}' not found in dataset.")
        position_keys.append(cleaned_positions_key)
        position_labels.append("Low conf cleaned")

    if aligned_positions_key is not None:
        if aligned_positions_key not in ds.keys():
            raise KeyError(f"Key '{aligned_positions_key}' not found in dataset.")
        position_keys.append(aligned_positions_key)
        position_labels.append("Aligned")

    if filtered_positions_key is not None:
        if filtered_positions_key not in ds.keys():
            raise KeyError(f"Key '{filtered_positions_key}' not found in dataset.")
        position_keys.append(filtered_positions_key)
        position_labels.append("Filtered")

    if scaled_positions_key is not None:
        if scaled_positions_key not in ds.keys():
            logger.info(f"Key '{scaled_positions_key}' not found in dataset.")
        position_keys.append(scaled_positions_key)
        position_labels.append("Scaled")

    # Load all position data
    positions_data = {}
    for key in position_keys:
        positions_data[key] = ds[key].values

    keypoints_labels = ds.keypoints.values

    if not frames:
        # Use the first position key to determine frame count
        first_key = position_keys[0]
        frames = [int(i * len(positions_data[first_key])) for i in [0.1, 0.3, 0.5, 0.7, 0.9]]

    num_frames = len(frames)
    num_cols = len(position_keys)

    # Create a figure with the appropriate number of columns
    fig, axes = plt.subplots(num_frames, num_cols, figsize=(6 * num_cols, 6 * num_frames))

    # Handle case where there's only one frame (axes would be 1D)
    if num_frames == 1:
        axes = axes.reshape(1, -1)

    # Define colors for each position type
    colors = ["blue", "orange", "green", "red", "purple"]

    # Get reference keypoint
    ref_keypoint = ds.centered_reference_keypoint
    ref_idx = np.where(keypoints_labels == ref_keypoint)[0][0]

    for i, frame in enumerate(frames):
        for j, (key, label) in enumerate(zip(position_keys, position_labels)):
            # Get position data for this frame
            x_pos = positions_data[key][frame, 0, :, 0]
            y_pos = positions_data[key][frame, 1, :, 0]

            # Get current axis
            ax = axes[i, j]

            # Identify keypoints that are NaN
            nan_keypoints = [
                keypoints_labels[k] for k in range(len(keypoints_labels)) if np.isnan(x_pos[k]) or np.isnan(y_pos[k])
            ]

            # Check if positions contain all NaNs
            if np.all(np.isnan(x_pos)) or np.all(np.isnan(y_pos)):
                ax.set_title(f"{label} - Frame {frame} (All NaNs)", fontsize=14, color="red")
                ax.axis("off")  # Hide axis since there is no data to plot
            else:
                margin = 10
                if scaled_positions_key and key == scaled_positions_key:
                    margin = 0.1
                x_min, x_max = np.nanmin(x_pos) - margin, np.nanmax(x_pos) + margin  # Add a margin
                y_min, y_max = np.nanmin(y_pos) - margin, np.nanmax(y_pos) + margin

                # Plot scatter points
                ax.scatter(x_pos, y_pos, c=colors[j % len(colors)], label=label)

                # Add keypoint labels
                for k, (x, y) in enumerate(zip(x_pos, y_pos)):
                    ax.text(x, y, keypoints_labels[k], fontsize=10, color=colors[j % len(colors)])

                # Include NaN keypoints in the title
                if nan_keypoints:
                    nan_text = ", ".join(nan_keypoints)
                    title_text = f"{label} - Frame {frame}\nNaNs: {nan_text}"
                else:
                    title_text = f"{label} - Frame {frame}"

                ax.set_title(title_text, fontsize=14)
                ax.set_xlabel("X", fontsize=12)
                ax.set_ylabel("Y", fontsize=12)

                # Draw reference lines
                if key == aligned_positions_key:
                    # For aligned positions, use (0,0) as reference
                    ax.axhline(0, color="gray", linestyle="--")
                    ax.axvline(0, color="gray", linestyle="--")
                else:
                    # For other positions, use the reference keypoint
                    ref_x = x_pos[ref_idx]
                    ref_y = y_pos[ref_idx]
                    ax.axhline(ref_y, color="gray", linestyle="--")
                    ax.axvline(ref_x, color="gray", linestyle="--")

                # Ensure square aspect by making the limits have equal range
                x_range = x_max - x_min
                y_range = y_max - y_min
                max_range = max(x_range, y_range)
                x_center = (x_max + x_min) / 2
                y_center = (y_max + y_min) / 2
                ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
                ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
                ax.set_aspect("equal")

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, top=0.9)  # Control spacing and add top margin for title
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
    sample_offset: int = 0,
    original_positions_key: str | None = "position",
    aligned_positions_key: str | None = "position_egocentric_aligned",
    filtered_positions_key: str | None = "position_processed",
    scaled_positions_key: str | None = "position_scaled",
    keypoints: list | None = None,
    save_to_file: bool = False,
    show_figure: bool = True,
):
    """
    Visualize the preprocessing results by plotting position data in a timeseries plot.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    session_index : int, optional
        Index of the session to visualize.
    n_samples : int, optional
        Number of samples to plot.
    sample_offset : int, optional
        Starting index for the time series data. Default is 0 (start from beginning).
    original_positions_key : str | None, optional
        Key for the original positions. If None, this position type will be skipped.
    aligned_positions_key : str | None, optional
        Key for the aligned positions. If None, this position type will be skipped.
    filtered_positions_key : str | None, optional
        Key for the filtered positions. If None, this position type will be skipped.
    scaled_positions_key : str | None, optional
        Key for the scaled positions. If None, this position type will be skipped.
    keypoints : list | None, optional
        List of keypoint names to include in the visualization. If None or empty list,
        all keypoints will be included.
    save_to_file : bool, optional
        Whether to save the figure to a file.
    show_figure : bool, optional
        Whether to show the figure.

    Returns
    -------
    None
    """
    import matplotlib

    if not show_figure:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_path = config["project_path"]
    sessions = config["session_names"]
    session = sessions[session_index]

    # Read session data
    file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
    _, _, ds = read_pose_estimation_file(file_path=file_path)

    # Create a list of position keys and labels, filtering out None values
    position_keys = []
    position_labels = []

    if original_positions_key is not None:
        if original_positions_key not in ds.keys():
            raise KeyError(f"Key '{original_positions_key}' not found in dataset.")
        position_keys.append(original_positions_key)
        position_labels.append("Original Allocentric")

    if aligned_positions_key is not None:
        if aligned_positions_key not in ds.keys():
            raise KeyError(f"Key '{aligned_positions_key}' not found in dataset.")
        position_keys.append(aligned_positions_key)
        position_labels.append("Aligned Egocentric")

    if filtered_positions_key is not None:
        if filtered_positions_key not in ds.keys():
            raise KeyError(f"Key '{filtered_positions_key}' not found in dataset.")
        position_keys.append(filtered_positions_key)
        position_labels.append("Filtered Egocentric")

    if scaled_positions_key is not None:
        if scaled_positions_key not in ds.keys():
            raise KeyError(f"Key '{scaled_positions_key}' not found in dataset.")
        position_keys.append(scaled_positions_key)
        position_labels.append("Scaled Egocentric")

    # Count how many position types we have
    num_positions = len(position_keys)

    if num_positions == 0:
        raise ValueError("No valid position keys provided.")

    # Create a grid of subplots (or fewer rows if fewer position types)
    fig, axes = plt.subplots(num_positions, 2, figsize=(18, 4 * num_positions))

    # Handle case where there's only one position type (axes would be 1D)
    if num_positions == 1:
        axes = axes.reshape(1, -1)

    individual = "individual_0"
    all_keypoints = ds.keypoints.values

    # Filter keypoints if a list is provided
    if keypoints is not None and len(keypoints) > 0:
        # Validate that all provided keypoints exist in the dataset
        invalid_keypoints = [kp for kp in keypoints if kp not in all_keypoints]
        if invalid_keypoints:
            raise ValueError(f"Invalid keypoint names: {', '.join(invalid_keypoints)}")
        keypoints_to_plot = keypoints
    else:
        keypoints_to_plot = all_keypoints

    # Create a colormap with distinguishable colors
    cmap = get_cmap("tab10") if len(keypoints_to_plot) <= 10 else get_cmap("tab20")
    colors = [cmap(i / len(keypoints_to_plot)) for i in range(len(keypoints_to_plot))]

    # Plot each position type
    for i, (key, label) in enumerate(zip(position_keys, position_labels)):
        # For each keypoint
        for j, kp in enumerate(keypoints_to_plot):
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

            # X coordinates in first column
            ds[key].sel(**sel_x)[sample_offset : sample_offset + n_samples].plot(
                linewidth=1.5,
                ax=axes[i, 0],
                label=kp,
                color=colors[j],
            )

            # Y coordinates in second column
            ds[key].sel(**sel_y)[sample_offset : sample_offset + n_samples].plot(
                linewidth=1.5,
                ax=axes[i, 1],
                label=kp,
                color=colors[j],
            )

    # Set titles and labels
    for i, label in enumerate(position_labels):
        axes[i, 0].set_title(f"{label} X", fontsize=14)
        axes[i, 1].set_title(f"{label} Y", fontsize=14)
        axes[i, 0].set_xlabel("Time", fontsize=12)
        axes[i, 1].set_xlabel("Time", fontsize=12)
        axes[i, 0].set_ylabel("Position", fontsize=12)
        axes[i, 1].set_ylabel("Position", fontsize=12)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.9)

    # Add a figure-level title
    fig.suptitle(
        f"{session} - Timeseries Visualization (Samples {sample_offset}-{sample_offset+n_samples})",
        fontsize=16,
        y=0.98,  # Position the title higher
    )

    # Add a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(5, len(labels)),
        bbox_to_anchor=(0.5, 0.95),
        fontsize=10,
    )

    if save_to_file:
        save_fig_path = Path(project_path) / "reports" / "figures" / f"{session}_preprocessing_timeseries.png"
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_fig_path))

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def visualize_preprocessing_cloud(
    config: dict,
    session_index: int = 0,
    n_samples: int = 1000,
    aligned_positions_key: str | None = "position_egocentric_aligned",
    filtered_positions_key: str | None = "position_processed",
    scaled_positions_key: str | None = "position_scaled",
    keypoints: list | None = None,
    alpha: float = 0.3,
    save_to_file: bool = False,
    show_figure: bool = True,
):
    """
    Visualize the preprocessing results by plotting a cloud of keypoint positions across multiple frames.
    Only includes aligned, filtered, and scaled positions as these are in comparable coordinate systems.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    session_index : int, optional
        Index of the session to visualize.
    n_samples : int, optional
        Number of frames to include in the visualization. Frames are randomly sampled.
    aligned_positions_key : str | None, optional
        Key for the egocentric aligned positions. If None, this position type will be skipped.
    filtered_positions_key : str | None, optional
        Key for the filtered positions. If None, this position type will be skipped.
    scaled_positions_key : str | None, optional
        Key for the scaled positions. If None, this position type will be skipped.
    keypoints : list | None, optional
        List of keypoint names to include. If None, all keypoints will be included.
    alpha : float, optional
        Transparency level for the dots (0.0 to 1.0).
    save_to_file : bool, optional
        Whether to save the figure to a file.
    show_figure : bool, optional
        Whether to show the figure.

    Returns
    -------
    None
    """
    import matplotlib

    if not show_figure:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_path = config["project_path"]
    sessions = config["session_names"]
    session = sessions[session_index]

    # Read session data
    file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
    _, _, ds = read_pose_estimation_file(file_path=file_path)

    # Create a list of position keys and labels, filtering out None values
    position_keys = []
    position_labels = []

    if aligned_positions_key is not None:
        if aligned_positions_key not in ds.keys():
            raise KeyError(f"Key '{aligned_positions_key}' not found in dataset.")
        position_keys.append(aligned_positions_key)
        position_labels.append("Aligned")

    if filtered_positions_key is not None:
        if filtered_positions_key not in ds.keys():
            raise KeyError(f"Key '{filtered_positions_key}' not found in dataset.")
        position_keys.append(filtered_positions_key)
        position_labels.append("Filtered")

    if scaled_positions_key is not None:
        if scaled_positions_key not in ds.keys():
            raise KeyError(f"Key '{scaled_positions_key}' not found in dataset.")
        position_keys.append(scaled_positions_key)
        position_labels.append("Scaled")

    # Count how many position types we have
    num_positions = len(position_keys)

    if num_positions == 0:
        raise ValueError("No valid position keys provided.")

    # Load all position data
    positions_data = {}
    for key in position_keys:
        positions_data[key] = ds[key].values

    # Get all keypoints
    all_keypoints = ds.keypoints.values

    # Filter keypoints if a list is provided
    if keypoints is not None and len(keypoints) > 0:
        # Validate that all provided keypoints exist in the dataset
        invalid_keypoints = [kp for kp in keypoints if kp not in all_keypoints]
        if invalid_keypoints:
            raise ValueError(f"Invalid keypoint names: {', '.join(invalid_keypoints)}")
        keypoints_to_plot = keypoints
    else:
        keypoints_to_plot = all_keypoints

    # Get keypoint indices
    keypoint_indices = [np.where(all_keypoints == kp)[0][0] for kp in keypoints_to_plot]

    # Create a colormap with distinguishable colors
    cmap = get_cmap("tab10") if len(keypoints_to_plot) <= 10 else get_cmap("tab20")
    colors = [cmap(i / len(keypoints_to_plot)) for i in range(len(keypoints_to_plot))]

    # Create a figure with a single row of subplots
    # Add extra height for the legend at the bottom
    fig, axes = plt.subplots(1, num_positions, figsize=(6 * num_positions, 6))

    # Handle case where there's only one position type (axes would be 0D)
    if num_positions == 1:
        axes = np.array([axes])

    # Get reference keypoint
    ref_keypoint = ds.centered_reference_keypoint
    ref_idx = np.where(all_keypoints == ref_keypoint)[0][0]

    # Ensure n_samples doesn't exceed available data and randomly sample frames
    first_key = position_keys[0]
    total_frames = len(positions_data[first_key])
    n_samples = min(n_samples, total_frames)

    # Randomly sample frame indices
    frame_indices = np.random.choice(total_frames, size=n_samples, replace=False)

    # Plot each position type
    for j, (key, label) in enumerate(zip(position_keys, position_labels)):
        ax = axes[j]

        # Initialize min/max values for axis limits
        x_min, x_max = float("inf"), float("-inf")
        y_min, y_max = float("inf"), float("-inf")

        # Plot each keypoint as a separate color
        for k, kp_idx in enumerate(keypoint_indices):
            # Extract x and y positions for this keypoint across all frames
            x_positions = []
            y_positions = []

            for frame in frame_indices:
                x_pos = positions_data[key][frame, 0, kp_idx, 0]
                y_pos = positions_data[key][frame, 1, kp_idx, 0]

                if not np.isnan(x_pos) and not np.isnan(y_pos):
                    x_positions.append(x_pos)
                    y_positions.append(y_pos)

            # Skip if no valid positions
            if not x_positions:
                continue

            # Update min/max values for axis limits
            x_min = min(x_min, min(x_positions))
            x_max = max(x_max, max(x_positions))
            y_min = min(y_min, min(y_positions))
            y_max = max(y_max, max(y_positions))

            # Plot the cloud of points for this keypoint
            ax.scatter(
                x_positions,
                y_positions,
                color=colors[k],
                alpha=alpha,
                label=keypoints_to_plot[k],
                s=10,  # Smaller point size for better cloud visualization
            )

        # Add margin to axis limits
        margin = 0.1 if key == scaled_positions_key else 10
        if x_min != float("inf"):  # Check if we have valid data
            x_min -= margin
            x_max += margin
            y_min -= margin
            y_max += margin

            # Draw reference lines
            if key == aligned_positions_key:
                # For aligned positions, use (0,0) as reference
                ax.axhline(0, color="gray", linestyle="--")
                ax.axvline(0, color="gray", linestyle="--")
            else:
                # For other positions, we need to calculate the average reference point
                ref_x_positions = []
                ref_y_positions = []

                for frame in frame_indices:
                    ref_x = positions_data[key][frame, 0, ref_idx, 0]
                    ref_y = positions_data[key][frame, 1, ref_idx, 0]

                    if not np.isnan(ref_x) and not np.isnan(ref_y):
                        ref_x_positions.append(ref_x)
                        ref_y_positions.append(ref_y)

                if ref_x_positions:
                    ref_x_avg = np.mean(ref_x_positions)
                    ref_y_avg = np.mean(ref_y_positions)
                    ax.axhline(ref_y_avg, color="gray", linestyle="--")
                    ax.axvline(ref_x_avg, color="gray", linestyle="--")

            # Ensure square aspect by making the limits have equal range
            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range, y_range)
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
            ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
            ax.set_aspect("equal")

            # Set titles and labels
            ax.set_title(f"{label}", fontsize=14)
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
        else:
            # No valid data for this position type
            ax.set_title(f"{label} (No valid data)", fontsize=14, color="red")
            ax.axis("off")  # Hide axis since there is no data to plot

    # Add a figure-level title
    fig.suptitle(
        f"{session} - Cloud Visualization ({n_samples} randomly sampled frames)",
        fontsize=16,
        y=0.98,  # Position the title higher
    )

    # Create a custom legend with non-transparent markers
    if num_positions > 0:
        # Get the labels from the first axis that has handles
        handles, labels = [], []
        for ax in axes:
            h, labels_list = ax.get_legend_handles_labels()
            if h:  # Only add if there are handles (some axes might have no data)
                handles, labels = h, labels_list
                break

        if handles:
            # Create new handles without transparency for the legend
            legend_handles = []
            for i, handle in enumerate(handles):
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color=colors[i],
                        markersize=8,
                        linestyle="None",
                    )
                )

            # Add the legend below the plots to avoid overlap
            fig.legend(
                legend_handles,
                labels,
                loc="upper center",
                ncol=min(5, len(labels)),
                bbox_to_anchor=(0.5, -0.02),
                fontsize=10,
            )

            # Adjust the figure layout to make room for the legend
            plt.subplots_adjust(bottom=0.02)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, top=0.9)

    # Don't use tight_layout as it can override our manual adjustments for the legend

    if save_to_file:
        save_fig_path = Path(project_path) / "reports" / "figures" / f"{session}_preprocessing_cloud.png"
        save_fig_path.parent.mkdir(parents=True, exist_ok=True)
        # Save with bbox_inches='tight' to ensure the legend is included
        plt.savefig(str(save_fig_path), bbox_inches="tight")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)
