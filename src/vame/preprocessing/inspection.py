from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from vame.io.load_poses import read_pose_estimation_file


def pose_estimation_inspection(
    config,
    read_from_variable: str = "position_raw",
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
            all_confidence_values = confidence.copy()
        else:
            all_confidence_values = np.concatenate(
                (all_confidence_values, confidence),
                axis=0,
            )

        for individual in individuals:
            for keypoint in keypoints:
                # Get confidence values for this keypoint and individual
                conf_series = confidence.sel(
                    individuals=individual,
                    keypoints=keypoint,
                ).values

    # Plot pose estimation inspection results
    plot_pose_estimation_inspection(
        config=config,
        confidence_array=all_confidence_values[:, 0, 0],
    )
    return ds


def plot_pose_estimation_inspection(
    config,
    confidence_array: np.ndarray,
) -> None:
    """
    Plot pose estimation inspection results.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    ds : xarray.Dataset
        Dataset containing the pose estimation data.

    Returns
    -------
    None
    """
    confidence_reference = config["pose_confidence"]

    # Histogram trace (probability histogram)
    hist = go.Histogram(
        x=confidence_array,
        nbinsx=50,                   # tweak or drop for automatic bin count
        histnorm="probability",      # makes bar heights sum to 1
        marker_line_width=.5,
        marker_line_color="black",
        marker_color="#8d93b5",
        name="confidence scores",
    )

    # Layout with vertical reference line (shape) and annotation
    layout = go.Layout(
        title="Confidence-score distribution",
        bargap=0.05,
        width=800,
        height=600,
        plot_bgcolor="white",
        xaxis=dict(
            title="confidence",
            range=[0, 1],
            tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            ticktext=["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"],
            showgrid=False,
            zeroline=True,
            zerolinecolor="black"
        ),
        yaxis=dict(
            title="probability",
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="black"
        ),
        shapes=[
            # vertical dashed line
            dict(
                type="line",
                x0=confidence_reference,
                x1=confidence_reference,
                y0=0,
                y1=1,
                yref="paper",  # full plot height
                line=dict(color="black", width=2, dash="dot")
            )
        ],
        annotations=[
            dict(
                x=confidence_reference,
                y=1.05,
                yref="paper",
                text=f"reference = {confidence_reference:.2f}",
                showarrow=False,
                font=dict(color="black", size=12, family="Arial")
            )
        ]
    )

    fig = go.Figure(data=[hist], layout=layout)
    fig.show()
