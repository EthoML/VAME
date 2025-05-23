import os
import umap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from typing import Optional

import plotly.graph_objects as go

from vame.util.cli import get_sessions_from_user_input
from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger


def umap_embedding(
    config: dict,
    session: str,
    model_name: str,
) -> np.ndarray:
    """
    Perform UMAP embedding for given file and parameters.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    session : str
        Session name.
    model_name : str
        Model name.

    Returns
    -------
    np.ndarray
        UMAP embedding.
    """
    reducer = umap.UMAP(
        n_components=2,
        min_dist=config["min_dist"],
        n_neighbors=config["n_neighbors"],
        random_state=config["random_state"],
    )
    logger.info(f"UMAP calculation for session {session}")
    folder = os.path.join(
        config["project_path"],
        "results",
        session,
        model_name,
    )
    latent_vector = np.load(os.path.join(folder, "latent_vectors.npy"))
    num_points = config["num_points"]
    if num_points > latent_vector.shape[0]:
        num_points = latent_vector.shape[0]
    logger.info(f"Embedding {num_points} data points...")
    embed = reducer.fit_transform(latent_vector[:num_points, :])
    np.save(
        os.path.join(folder, "umap_embedding.npy"),
        embed,
    )
    return embed


def umap_vis(
    embed: np.ndarray,
    num_points: int = 30_000,
    labels: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
    label_type: str = "none",
) -> Figure:
    """
    Visualize UMAP embedding.

    Parameters
    ----------
    embed : np.ndarray
        UMAP embedding.
    num_points : int
        Number of data points to visualize. Default is 30,000.
    labels : np.ndarray, optional
        Motif or community labels. Default is None.
    title : str, optional
        Title for the plot. Default is None.
    show_legend : bool, optional
        Whether to show legend for labeled plots. Default is True.
    label_type : str, optional
        Type of labels ('none', 'motif', 'community'). Default is 'none'.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Randomly sample up to num_points rows without replacement
    n_samples = min(num_points, embed.shape[0])
    if embed.shape[0] > n_samples:
        indices = np.random.choice(embed.shape[0], size=n_samples, replace=False)
    else:
        indices = np.arange(n_samples)
    scatter_kwargs = {
        "x": embed[indices, 0],
        "y": embed[indices, 1],
        "s": 2,
        "alpha": 0.5,
    }
    if labels is not None:
        labels = np.array(labels)
        scatter_kwargs["c"] = labels[indices]
        scatter_kwargs["cmap"] = "Spectral"
        scatter_kwargs["alpha"] = 0.7

    plt.close("all")
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(**scatter_kwargs)
    plt.gca().set_aspect("equal", "datalim")
    plt.grid(False)

    # Add title if provided
    if title:
        plt.title(title, fontsize=14, fontweight="bold")

    # Add legend for labeled plots
    if labels is not None and show_legend:
        unique_labels = np.unique(labels[indices])
        if len(unique_labels) <= 20:  # Only show legend if not too many labels
            if label_type == "motif":
                legend_title = "Motif"
            elif label_type == "community":
                legend_title = "Community"
            else:
                legend_title = "Label"

            # Create legend with discrete colors
            handles = []
            for label_val in sorted(unique_labels):
                color = cm.Spectral(label_val / max(unique_labels) if max(unique_labels) > 0 else 0)
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=8,
                        label=f"{legend_title} {int(label_val)}",
                    )
                )

            plt.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.tight_layout()

    return fig


def visualize_umap(
    config: dict,
    num_points: int = 30_000,
    save_to_file: bool = True,
    show_figure: bool = True,
    save_logs: bool = True,
) -> None:
    """
    Visualize UMAP embeddings based on configuration settings.
    Fills in the values in the "visualization_umap" key of the states.json file.
    Saves results files at:
    - project_name/
        - results/
            - file_name/
                - model_name/
                    - segmentation_algorithm-n_clusters/
                        - community/
                            - umap_embedding_file_name.npy
                            - umap_vis_label_none_file_name.png  (UMAP visualization without labels)
                            - umap_vis_motif_file_name.png  (UMAP visualization with motif labels)
                            - umap_vis_community_file_name.png  (UMAP visualization with community labels)

    Parameters
    ----------
    config : dict
        Configuration parameters.
    num_points : int, optional
        Number of data points to visualize. Default is 30,000.
    save_to_file : bool, optional
        Save the figure to file. Default is True.
    show_figure : bool, optional
        Show the figure. Default is True.
    save_logs : bool, optional
        Save logs. Default is True.

    Returns
    -------
    None
    """
    try:
        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "report.log"
            logger_config.add_file_handler(str(log_path))

        model_name = config["model_name"]
        n_clusters = config["n_clusters"]
        segmentation_algorithms = config["segmentation_algorithms"]

        # Get sessions
        if config["all_data"] in ["Yes", "yes", "True", "true", True]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                config=config,
                action_message="generate visualization",
            )

        save_path_base = Path(config["project_path"]) / "reports" / "umap"
        if not save_path_base.exists():
            os.makedirs(save_path_base)

        all_sessions_embeddings = []
        all_sessions_labels_motif = {}
        all_sessions_labels_community = {}
        for session in sessions:
            base_path = Path(config["project_path"]) / "results" / session / model_name
            umap_embeddings_path = base_path / "umap_embedding.npy"
            if umap_embeddings_path.exists():
                logger.info(f"UMAP embedding already exists for session {session}")
                embed = np.load(str(umap_embeddings_path.resolve()))
            else:
                logger.info(f"Computing UMAP embedding for session {session}")
                embed = umap_embedding(
                    config=config,
                    session=session,
                    model_name=model_name,
                )
            all_sessions_embeddings.append(embed)
            # all_sessions_labels_session_name.append([session] * len(embed))
            for seg in segmentation_algorithms:
                labels_names = ["none", "motif", "community"]
                labels_motif = []
                labels_community = []
                for label in labels_names:
                    if label == "none":
                        # output_figure_file_name = f"umap_{session}_{model_name}_{seg}-{n_clusters}.png"
                        labels = None
                    elif label == "motif":
                        # output_figure_file_name = f"umap_{session}_{model_name}_{seg}-{n_clusters}_motif.png"
                        labels_file_path = (
                            base_path / f"{seg}-{n_clusters}" / f"{n_clusters}_{seg}_label_{session}.npy"
                        )
                        if labels_file_path.exists():
                            labels = np.load(str(labels_file_path.resolve()))
                            labels_motif.append(labels)
                        else:
                            logger.warning(f"Motif labels not found for session {session}. Skipping visualization.")
                            continue
                    elif label == "community":
                        # output_figure_file_name = f"umap_{session}_{model_name}_{seg}-{n_clusters}_community.png"
                        labels_file_path = (
                            base_path / f"{seg}-{n_clusters}" / "community" / f"cohort_community_label_{session}.npy"
                        )
                        if labels_file_path.exists():
                            labels = np.load(str(labels_file_path.resolve()))
                            labels_community.append(labels)
                        else:
                            logger.warning(
                                f"Community labels not found for session {session}. Skipping visualization."
                            )
                            continue

                all_sessions_labels_motif[f"{seg}-{n_clusters}"] = np.concatenate(labels_motif, axis=0)
                all_sessions_labels_community[f"{seg}-{n_clusters}"] = np.concatenate(labels_community, axis=0)

        all_embeddings = np.concatenate(all_sessions_embeddings, axis=0)

        # Generate UMAP figures
        for seg in segmentation_algorithms:
            for label in labels_names:
                if label == "none":
                    output_figure_file_name = f"umap_{model_name}_{seg}-{n_clusters}.png"
                    labels = None
                elif label == "motif":
                    output_figure_file_name = f"umap_{model_name}_{seg}-{n_clusters}_motif.png"
                    labels = all_sessions_labels_motif[f"{seg}-{n_clusters}"]
                elif label == "community":
                    output_figure_file_name = f"umap_{model_name}_{seg}-{n_clusters}_community.png"
                    labels = all_sessions_labels_community[f"{seg}-{n_clusters}"]

                # Generate title
                if label == "none":
                    title = f"UMAP Visualization - Model: {model_name} | {seg}-{n_clusters}"
                elif label == "motif":
                    title = f"UMAP Visualization - Model: {model_name} | {seg}-{n_clusters} | Motif Labels"
                elif label == "community":
                    title = f"UMAP Visualization - Model: {model_name} | {seg}-{n_clusters} | Community Labels"

                fig = umap_vis(
                    embed=all_embeddings,
                    num_points=num_points,
                    labels=labels,
                    title=title,
                    show_legend=True,
                    label_type=label,
                )

                if save_to_file:
                    fig_path = save_path_base / output_figure_file_name
                    fig.savefig(fig_path)
                    logger.info(f"UMAP figure saved to {fig_path}")

                if show_figure:
                    plt.show()
                else:
                    plt.close(fig)

        # Generate interactive Plotly UMAP figures
        for seg in segmentation_algorithms:
            motif_labels = all_sessions_labels_motif[f"{seg}-{n_clusters}"]
            community_labels = all_sessions_labels_community[f"{seg}-{n_clusters}"]
            interactive_fig = umap_vis_plotly(
                embed=all_embeddings,
                labels_motif=motif_labels,
                labels_community=community_labels,
                num_points=num_points,
            )
            if save_to_file:
                html_path = save_path_base / f"umap_{model_name}_{seg}-{n_clusters}_interactive.html"
                interactive_fig.write_html(str(html_path))
                logger.info(f"Interactive UMAP figure saved to {html_path}")
            if show_figure:
                interactive_fig.show()

    except Exception as e:
        logger.exception(str(e))
        raise e
    finally:
        logger_config.remove_file_handler()


def umap_vis_plotly(
    embed: np.ndarray,
    labels_motif: Optional[np.ndarray] = None,
    labels_community: Optional[np.ndarray] = None,
    num_points: int = 30_000,
) -> go.Figure:
    """
    Create an interactive Plotly UMAP scatter with dropdown to select labels:
      - None: grey points
      - Motif: colored by motif labels
      - Community: colored by community labels

    Parameters
    ----------
    embed : np.ndarray
        2D UMAP embedding array of shape (N,2).
    labels_motif : np.ndarray or None
        1D motif labels of length N.
    labels_community : np.ndarray or None
        1D community labels of length N.
    num_points : int
        Maximum number of points to show.

    Returns
    -------
    plotly.graph_objs.Figure
        The interactive Plotly figure.
    """
    # Randomly sample up to num_points rows without replacement
    n_samples = min(num_points, embed.shape[0])
    if embed.shape[0] > n_samples:
        indices = np.random.choice(embed.shape[0], size=n_samples, replace=False)
    else:
        indices = np.arange(n_samples)
    x_vals = embed[indices, 0]
    y_vals = embed[indices, 1]

    # Trace for no labeling (grey)
    trace_none = go.Scattergl(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(color="grey", size=4, opacity=0.6),
        name="None",
        visible=True,
    )
    data = [trace_none]

    # Trace for motif labels
    if labels_motif is not None:
        motif_vals = np.array(labels_motif)[indices]
        trace_motif = go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(color=motif_vals, colorscale="Spectral", size=4, opacity=0.6),
            name="Motif",
            visible=False,
        )
        data.append(trace_motif)

    # Trace for community labels
    if labels_community is not None:
        comm_vals = np.array(labels_community)[indices]
        trace_comm = go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(color=comm_vals, colorscale="Viridis", size=4, opacity=0.6),
            name="Community",
            visible=False,
        )
        data.append(trace_comm)

    # Create dropdown buttons
    buttons = [
        dict(
            label="None",
            method="update",
            args=[{"visible": [True] + [False] * (len(data) - 1)}, {"title": "UMAP - None"}],
        )
    ]
    if labels_motif is not None:
        visible = [False] * len(data)
        visible[1] = True
        buttons.append(
            dict(label="Motif", method="update", args=[{"visible": visible}, {"title": "UMAP - Motif"}]),
        )
    if labels_community is not None:
        idx_comm = 2 if labels_motif is not None else 1
        visible = [False] * len(data)
        visible[idx_comm] = True
        buttons.append(
            dict(
                label="Community",
                method="update",
                args=[{"visible": visible}, {"title": "UMAP - Community"}],
            ),
        )

    updatemenus = [dict(active=0, buttons=buttons, x=1.1, y=1)]
    layout = go.Layout(
        title="UMAP - None",
        xaxis=dict(title="UMAP 1"),
        yaxis=dict(title="UMAP 2"),
        updatemenus=updatemenus,
        margin=dict(l=40, r=40, t=40, b=40),
        height=600,
        width=800,
    )
    fig = go.Figure(data=data, layout=layout)
    return fig
