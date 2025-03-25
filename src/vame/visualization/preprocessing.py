from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

from vame.io.load_poses import read_pose_estimation_file


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
            raise KeyError(f"Key '{scaled_positions_key}' not found in dataset.")
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
                keypoints_labels[k] for k in range(len(keypoints_labels))
                if np.isnan(x_pos[k]) or np.isnan(y_pos[k])
            ]

            # Check if positions contain all NaNs
            if np.all(np.isnan(x_pos)) or np.all(np.isnan(y_pos)):
                ax.set_title(f"{label} - Frame {frame} (All NaNs)", fontsize=14, color="red")
                ax.axis("off")  # Hide axis since there is no data to plot
            else:
                margin = 10
                if scaled_positions_key and key == scaled_positions_key:
                    margin = .1
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
                ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
                ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
                ax.set_aspect('equal')

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
