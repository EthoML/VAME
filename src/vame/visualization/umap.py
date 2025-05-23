import os
import umap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from typing import Optional, Literal

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
    show_figure: Literal["none", "matplotlib", "plotly", "all"] = "none",
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
    show_figure : Literal["none", "matplotlib", "plotly", "all"], optional
        Show the figure. Default is "none".
        - "none": do not show
        - "matplotlib": show with matplotlib
        - "plotly": show with plotly
        - "all": show with both matplotlib and plotly
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

                if show_figure in ["matplotlib", "all"]:
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
                title=f"UMAP Visualization - Model: {model_name} | {seg}-{n_clusters}",
            )
            config_plotly = {"displaylogo": False, "scrollZoom": True}
            if save_to_file:
                html_path = save_path_base / f"umap_{model_name}_{seg}-{n_clusters}_interactive.html"
                interactive_fig.write_html(str(html_path), config=config_plotly)
                logger.info(f"Interactive UMAP figure saved to {html_path}")
            if show_figure in ["plotly", "all"]:
                interactive_fig.show(config=config_plotly)
                return interactive_fig

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
    title: str = "UMAP",
    marker_size: float = 3.5,
    marker_opacity: float = 0.7,
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
    title : str
        Title for the figure. Defaults to "UMAP".
    marker_size : float
        Size of the markers in the plot.
    marker_opacity : float
        Opacity of the markers in the plot.

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
        marker=dict(
            color="grey",
            size=marker_size,
            opacity=marker_opacity,
        ),
        name="None",
        visible=True,
    )
    data = [trace_none]

    # Trace for motif labels - create separate traces for each unique motif
    if labels_motif is not None:
        motif_vals = np.array(labels_motif)[indices]
        unique_motifs = np.unique(motif_vals)

        # Get colors from Spectral colorscale
        import plotly.colors as pc
        spectral_colors = pc.sample_colorscale("Spectral", [i/(len(unique_motifs)-1) if len(unique_motifs) > 1 else 0.5 for i in range(len(unique_motifs))])

        for i, motif_id in enumerate(unique_motifs):
            mask = motif_vals == motif_id
            trace_motif = go.Scattergl(
                x=x_vals[mask],
                y=y_vals[mask],
                mode="markers",
                marker=dict(
                    color=spectral_colors[i],
                    size=marker_size,
                    opacity=marker_opacity,
                ),
                name=f"Motif {int(motif_id)}",
                visible=False,
            )
            data.append(trace_motif)

    # Trace for community labels - create separate traces for each unique community
    if labels_community is not None:
        comm_vals = np.array(labels_community)[indices]
        unique_communities = np.unique(comm_vals)

        # Get colors from Viridis colorscale
        import plotly.colors as pc
        viridis_colors = pc.sample_colorscale("Viridis", [i/(len(unique_communities)-1) if len(unique_communities) > 1 else 0.5 for i in range(len(unique_communities))])

        for i, comm_id in enumerate(unique_communities):
            mask = comm_vals == comm_id
            trace_comm = go.Scattergl(
                x=x_vals[mask],
                y=y_vals[mask],
                mode="markers",
                marker=dict(
                    color=viridis_colors[i],
                    size=marker_size,
                    opacity=marker_opacity,
                ),
                name=f"Community {int(comm_id)}",
                visible=False,
            )
            data.append(trace_comm)

    # Create dropdown buttons - need to update visibility masks for multiple traces per group
    mask_none = [True] + [False] * (len(data) - 1)

    # For motif: show all motif traces (indices 1 to 1+num_motifs-1)
    mask_motif = [False] * len(data)
    if labels_motif is not None:
        unique_motifs = np.unique(np.array(labels_motif)[indices])
        for i in range(len(unique_motifs)):
            mask_motif[1 + i] = True

    # For community: show all community traces (after motif traces)
    mask_comm = [False] * len(data)
    if labels_community is not None:
        unique_communities = np.unique(np.array(labels_community)[indices])
        start_idx = 1 + (len(unique_motifs) if labels_motif is not None else 0)
        for i in range(len(unique_communities)):
            mask_comm[start_idx + i] = True

    buttons = [
        dict(label="None", method="restyle", args=["visible", mask_none]),
    ]
    if labels_motif is not None:
        buttons.append(
            dict(label="Motif", method="restyle", args=["visible", mask_motif]),
        )
    if labels_community is not None:
        buttons.append(
            dict(label="Community", method="restyle", args=["visible", mask_comm]),
        )

    updatemenus = [dict(active=0, buttons=buttons, x=0., y=1., xanchor="left", yanchor="bottom")]
    layout = go.Layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(
            title=dict(text="UMAP 1", font=dict(size=16)),
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=dict(text="UMAP 2", font=dict(size=16)),
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
            tickfont=dict(size=14),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        shapes=[
            dict(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1, line=dict(color="black", width=1)),
            dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0, line=dict(color="black", width=1)),
        ],
        legend=dict(
            title=dict(text="Label", font=dict(size=16)),
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        updatemenus=updatemenus,
        margin=dict(l=40, r=200, t=90, b=40),
        height=800,
        width=1100,
        dragmode="pan",
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False),
    )
    return fig
