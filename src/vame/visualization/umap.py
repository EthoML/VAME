from typing import Optional, Literal
import os
import umap
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import plotly.graph_objects as go

from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger


def umap_embedding(
    config: dict,
    num_points: int = 30_000,
) -> xr.Dataset:
    """
    Perform UMAP embedding for a sample of the entire project.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    num_points : int
        Number of points to use for UMAP embedding. Default is 30,000.

    Returns
    -------
    xr.Dataset
        UMAP embedding and accompanying data for the sampled indices.
    """
    model_name = config["model_name"]
    segmentation_algorithms = config["segmentation_algorithms"]
    n_clusters = config["n_clusters"]

    # Load and concatenate all sessions latent vectors
    latent_vectors_list = []
    all_session_names = []
    for session in config["session_names"]:
        # Gather all latent vectors
        latent_vectors_path = (
            Path(config["project_path"])
            / "results"
            / session
            / model_name
            / "latent_vectors.npy"
        )
        if not latent_vectors_path.exists():
            raise ValueError(
                f"Latent space data not found at: {latent_vectors_path}. "
                "Make sure to run vame.segment_session() first."
            )
        latent_data = np.load(latent_vectors_path)
        latent_vectors_list.append(latent_data)

        # Gather all session names
        all_session_names.extend([session] * len(latent_data))

    # Concatenate all latent vectors at once
    all_latent_vectors = np.concatenate(latent_vectors_list, axis=0)

    # Randomly sample up to num_points rows without replacement
    if all_latent_vectors.shape[0] > num_points:
        indices = np.random.choice(
            all_latent_vectors.shape[0],
            size=num_points,
            replace=False,
        )
        indices = np.sort(indices)
    else:
        indices = np.arange(all_latent_vectors.shape[0])
    all_latent_vectors_selected = all_latent_vectors[indices]
    all_session_names_selected = np.array(all_session_names)[indices]

    # Run UMAP
    logger.info("Running UMAP calculation...")
    reducer = umap.UMAP(
        n_components=2,
        min_dist=config["min_dist"],
        n_neighbors=config["n_neighbors"],
        random_state=config["random_state"],
    )
    umap_embeddings = reducer.fit_transform(all_latent_vectors_selected)

    # Gather motifs and communities aligned with selected points for UMAP
    all_motifs = {
        "hmm": np.array([]),
        "kmeans": np.array([]),
    }
    all_communities = {
        "hmm": np.array([]),
        "kmeans": np.array([]),
    }
    for session in config["session_names"]:
        for seg in segmentation_algorithms:
            # Gather all motifs
            motifs_path = (
                Path(config["project_path"])
                / "results"
                / session
                / model_name
                / f"{seg}-{n_clusters}"
                / f"{n_clusters}_{seg}_label_{session}.npy"
            )
            if motifs_path.exists():
                m = np.load(motifs_path)
                all_motifs[seg] = np.concatenate((all_motifs[seg], m), axis=0)
            else:
                logger.warning(
                    f"Motif labels not found for session {session}, "
                    f"segmentation {seg}-{n_clusters}. "
                    "Motif info will not be available for UMAP visualization."
                )
                continue

            # Gather all communities
            community_path = (
                Path(config["project_path"])
                / "results"
                / session
                / model_name
                / f"{seg}-{n_clusters}"
                / "community"
                / f"cohort_community_label_{session}.npy"
            )
            if community_path.exists():
                c = np.load(community_path)
                all_communities[seg] = np.concatenate(
                    (all_communities[seg], c), axis=0
                )
            else:
                logger.warning(
                    f"Community labels not found for session {session}, "
                    f"segmentation {seg}-{n_clusters}. "
                    "Community info will not be available for UMAP visualization."
                )
                continue

    if len(all_motifs["hmm"]) > 0:
        motifs_selected_hmm = all_motifs["hmm"][indices]
    else:
        motifs_selected_hmm = np.array([])
    if len(all_motifs["kmeans"]) > 0:
        motifs_selected_kmeans = all_motifs["kmeans"][indices]
    else:
        motifs_selected_kmeans = np.array([])
    if len(all_communities["hmm"]) > 0:
        communities_selected_hmm = all_communities["hmm"][indices]
    else:
        communities_selected_hmm = np.array([])
    if len(all_communities["kmeans"]) > 0:
        communities_selected_kmeans = all_communities["kmeans"][indices]
    else:
        communities_selected_kmeans = np.array([])

    # Build an xarray.Dataset
    ds = xr.Dataset(
        data_vars={
            "umap_embeddings": (("points", "components"), umap_embeddings),
            "session_names": ("points", all_session_names_selected.astype(str)),
            "motifs_hmm": ("points", motifs_selected_hmm),
            "motifs_kmeans": ("points", motifs_selected_kmeans),
            "communities_hmm": ("points", communities_selected_hmm),
            "communities_kmeans": ("points", communities_selected_kmeans),
        },
        coords={
            "points": indices,
            "components": ["UMAP_1", "UMAP_2"],
        },
        attrs={
            "num_points_in_project": int(all_latent_vectors.shape[0]),
            "num_points_selected": int(len(indices)),
            "random_state": int(config["random_state"]),
        },
    )

    # Save as netCDF (NETCDF4/HDF5 container)
    nc_path = Path(config["project_path"]) / "results" / "umap_embedding.nc"
    ds.to_netcdf(
        path=nc_path,
        format="NETCDF4",
        encoding={
            "umap_embeddings": {"zlib": True, "complevel": 4},
            "session_names": {"dtype": "S1"},  # store strings as char arrays
        },
    )
    logger.info(f"UMAP embeddings saved to {nc_path}")
    return ds


def umap_vis_matplotlib(
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
        scatter_kwargs["cmap"] = "hsv"
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
                color = cm.get_cmap('hsv')(
                    label_val / max(unique_labels) if max(unique_labels) > 0 else 0
                )
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

            plt.legend(
                handles=handles,
                title=legend_title,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )

    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.tight_layout()

    return fig


def umap_vis_plotly(
    embed: np.ndarray,
    labels_motif: Optional[np.ndarray] = None,
    labels_community: Optional[np.ndarray] = None,
    session_names: Optional[list] = None,
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
      - Session: colored by session labels

    Parameters
    ----------
    embed : np.ndarray
        2D UMAP embedding array of shape (N,2).
    labels_motif : np.ndarray or None
        1D motif labels of length N.
    labels_community : np.ndarray or None
        1D community labels of length N.
    session_names : list or None
        List of session names for each point.
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
    n_samples = min(num_points, embed.shape[0])
    if embed.shape[0] > n_samples:
        indices = np.random.choice(embed.shape[0], size=n_samples, replace=False)
    else:
        indices = np.arange(embed.shape[0])

    x_vals = embed[indices, 0]
    y_vals = embed[indices, 1]

    # Prepare hover data
    if session_names is not None:
        session_vals = [session_names[i] for i in indices]
    else:
        session_vals = ["N/A"] * len(indices)

    # Prepare motif and community values for hover
    if labels_motif is not None:
        motif_vals = np.array(labels_motif)[indices]
    else:
        motif_vals = ["N/A"] * len(indices)

    if labels_community is not None:
        comm_vals = np.array(labels_community)[indices]
    else:
        comm_vals = ["N/A"] * len(indices)

    # Custom hover template (without timestamp)
    hover_template = (
        "<b>Session:</b> %{customdata[0]}<br>"
        "<b>Motif:</b> %{customdata[1]}<br>"
        "<b>Community:</b> %{customdata[2]}<br>"
        "<extra></extra>"
    )

    # Trace for no labeling (grey)
    customdata_none = list(zip(session_vals, motif_vals, comm_vals))
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
        customdata=customdata_none,
        hovertemplate=hover_template,
    )
    data = [trace_none]

    # Trace for motif labels - create separate traces for each unique motif
    if labels_motif is not None:
        motif_vals = np.array(labels_motif)[indices]
        unique_motifs = np.unique(motif_vals)

        # Get colors from HSV colorscale
        import plotly.colors as pc

        hsv_colors = pc.sample_colorscale(
            "HSV",
            [
                i / (len(unique_motifs) - 1) if len(unique_motifs) > 1 else 0.5
                for i in range(len(unique_motifs))
            ],
        )

        for i, motif_id in enumerate(unique_motifs):
            mask = motif_vals == motif_id
            # Prepare customdata for this motif
            customdata_motif = [
                customdata_none[j] for j in range(len(mask)) if mask[j]
            ]
            trace_motif = go.Scattergl(
                x=x_vals[mask],
                y=y_vals[mask],
                mode="markers",
                marker=dict(
                    color=hsv_colors[i],
                    size=marker_size,
                    opacity=marker_opacity,
                ),
                name=f"Motif {int(motif_id)}",
                visible=False,
                showlegend=True,
                customdata=customdata_motif,
                hovertemplate=hover_template,
            )
            data.append(trace_motif)

    # Trace for community labels - create separate traces for each unique community
    if labels_community is not None:
        comm_vals = np.array(labels_community)[indices]
        unique_communities = np.unique(comm_vals)

        # Get colors from HSV colorscale
        import plotly.colors as pc

        hsv_colors_comm = pc.sample_colorscale(
            "HSV",
            [
                i / (len(unique_communities) - 1) if len(unique_communities) > 1 else 0.5
                for i in range(len(unique_communities))
            ],
        )

        for i, comm_id in enumerate(unique_communities):
            mask = comm_vals == comm_id
            # Prepare customdata for this community
            customdata_comm = [
                customdata_none[j] for j in range(len(mask)) if mask[j]
            ]
            trace_comm = go.Scattergl(
                x=x_vals[mask],
                y=y_vals[mask],
                mode="markers",
                marker=dict(
                    color=hsv_colors_comm[i],
                    size=marker_size,
                    opacity=marker_opacity,
                ),
                name=f"Community {int(comm_id)}",
                visible=False,
                showlegend=True,
                customdata=customdata_comm,
                hovertemplate=hover_template,
            )
            data.append(trace_comm)

    # Trace for session labels - create separate traces for each unique session
    if session_names is not None:
        session_vals = np.array(session_vals)
        unique_sessions = np.unique(session_vals)

        # Get colors from HSV colorscale
        import plotly.colors as pc

        hsv_colors_sess = pc.sample_colorscale(
            "HSV",
            [
                i / (len(unique_sessions) - 1) if len(unique_sessions) > 1 else 0.5
                for i in range(len(unique_sessions))
            ],
        )

        for i, session_id in enumerate(unique_sessions):
            mask = session_vals == session_id
            # Prepare customdata for this session
            customdata_sess = [
                customdata_none[j] for j in range(len(mask)) if mask[j]
            ]
            trace_sess = go.Scattergl(
                x=x_vals[mask],
                y=y_vals[mask],
                mode="markers",
                marker=dict(
                    color=hsv_colors_sess[i],
                    size=marker_size,
                    opacity=marker_opacity,
                ),
                name=f"Session {session_id}",
                visible=False,
                showlegend=True,
                customdata=customdata_sess,
                hovertemplate=hover_template,
            )
            data.append(trace_sess)

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

    # For session: show all session traces (after community traces)
    mask_sess = [False] * len(data)
    if session_names is not None:
        unique_sessions = np.unique(np.array(session_vals))
        start_idx = 1
        if labels_motif is not None:
            start_idx += len(np.unique(np.array(labels_motif)[indices]))
        if labels_community is not None:
            start_idx += len(np.unique(np.array(labels_community)[indices]))
        for i in range(len(unique_sessions)):
            mask_sess[start_idx + i] = True

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
    if session_names is not None:
        buttons.append(
            dict(label="Session", method="restyle", args=["visible", mask_sess]),
        )

    updatemenus = [
        dict(active=0, buttons=buttons, x=0.98, y=1.0, xanchor="left", yanchor="bottom")
    ]
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
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=0,
                x1=0,
                y0=0,
                y1=1,
                line=dict(color="black", width=1),
            ),
            dict(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=0,
                y1=0,
                line=dict(color="black", width=1),
            ),
        ],
        legend=dict(
            title=dict(text="Label", font=dict(size=16)),
            font=dict(size=14),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            itemsizing="constant",
            itemwidth=30,
        ),
        updatemenus=updatemenus,
        margin=dict(l=40, r=200, t=80, b=40),
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
            - umap_embedding.nc
        - reports/
            - umap/
                - umap_<model>_<seg>-<n_clusters>.png              (UMAP visualization without labels)
                - umap_<model>_<seg>-<n_clusters>_motif.png        (UMAP visualization with motif labels)
                - umap_<model>_<seg>-<n_clusters>_community.png    (UMAP visualization with community labels)
                - umap_<model>_<seg>-<n_clusters>_interactive.html (Interactive UMAP visualization)

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

        save_path_base = Path(config["project_path"]) / "reports" / "umap"
        if not save_path_base.exists():
            os.makedirs(save_path_base)

        # Check if the UMAP embedding already exists, if not, compute it
        base_path = Path(config["project_path"]) / "results"
        umap_embeddings_path = base_path / "umap_embedding.nc"
        if umap_embeddings_path.exists():
            logger.info(
                f"UMAP embedding already exists at {umap_embeddings_path}. Loading..."
            )
            umap_ds = xr.open_dataset(umap_embeddings_path)
        else:
            logger.info("Computing UMAP embeddings for all sessions combined...")
            umap_ds = umap_embedding(config=config, num_points=num_points)

        # Extract data from xarray dataset
        embeddings = umap_ds.umap_embeddings.values
        session_names = umap_ds.session_names.values
        motifs_hmm = (
            umap_ds.motifs_hmm.values if len(umap_ds.motifs_hmm.values) > 0 else None
        )
        motifs_kmeans = (
            umap_ds.motifs_kmeans.values
            if len(umap_ds.motifs_kmeans.values) > 0
            else None
        )
        communities_hmm = (
            umap_ds.communities_hmm.values
            if len(umap_ds.communities_hmm.values) > 0
            else None
        )
        communities_kmeans = (
            umap_ds.communities_kmeans.values
            if len(umap_ds.communities_kmeans.values) > 0
            else None
        )

        # Create label dictionaries organized by segmentation algorithm
        motif_labels = {}
        community_labels = {}
        for seg in segmentation_algorithms:
            if seg == "hmm":
                motif_labels[f"{seg}-{n_clusters}"] = motifs_hmm
                community_labels[f"{seg}-{n_clusters}"] = communities_hmm
            elif seg == "kmeans":
                motif_labels[f"{seg}-{n_clusters}"] = motifs_kmeans
                community_labels[f"{seg}-{n_clusters}"] = communities_kmeans

        # Define label types
        labels_names = ["none", "motif", "community"]

        # Generate UMAP figures
        for seg in segmentation_algorithms:
            for label in labels_names:
                if label == "none":
                    output_figure_file_name = f"umap_{model_name}_{seg}-{n_clusters}.png"
                    labels = None
                elif label == "motif":
                    output_figure_file_name = (
                        f"umap_{model_name}_{seg}-{n_clusters}_motif.png"
                    )
                    labels = motif_labels[f"{seg}-{n_clusters}"]
                elif label == "community":
                    output_figure_file_name = (
                        f"umap_{model_name}_{seg}-{n_clusters}_community.png"
                    )
                    labels = community_labels[f"{seg}-{n_clusters}"]

                # Skip if labels are None or empty for motif/community plots
                if label in ["motif", "community"] and (
                    labels is None or len(labels) == 0
                ):
                    logger.warning(
                        f"Skipping {label} visualization for {seg}-{n_clusters} - "
                        "no labels available"
                    )
                    continue

                # Generate title
                if label == "none":
                    title = (
                        f"UMAP Visualization - Model: {model_name} | {seg}-{n_clusters}"
                    )
                elif label == "motif":
                    title = (
                        f"UMAP Visualization - Model: {model_name} | "
                        f"{seg}-{n_clusters} | Motif Labels"
                    )
                elif label == "community":
                    title = (
                        f"UMAP Visualization - Model: {model_name} | "
                        f"{seg}-{n_clusters} | Community Labels"
                    )

                fig = umap_vis_matplotlib(
                    embed=embeddings,
                    num_points=len(
                        embeddings
                    ),  # Use all points since we already selected them
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
            motif_labels_seg = motif_labels[f"{seg}-{n_clusters}"]
            community_labels_seg = community_labels[f"{seg}-{n_clusters}"]

            # Skip if both motif and community labels are missing
            if (motif_labels_seg is None or len(motif_labels_seg) == 0) and (
                community_labels_seg is None or len(community_labels_seg) == 0
            ):
                logger.warning(
                    f"Skipping interactive visualization for {seg}-{n_clusters} - "
                    "no labels available"
                )
                continue

            interactive_fig = umap_vis_plotly(
                embed=embeddings,
                labels_motif=motif_labels_seg,
                labels_community=community_labels_seg,
                session_names=(
                    session_names.tolist() if session_names is not None else None
                ),
                num_points=len(
                    embeddings
                ),  # Use all points since we already selected them
                title=f"UMAP Visualization - Model: {model_name} | {seg}-{n_clusters}",
            )
            config_plotly = {"displaylogo": False, "scrollZoom": True}
            if save_to_file:
                html_path = (
                    save_path_base
                    / f"umap_{model_name}_{seg}-{n_clusters}_interactive.html"
                )
                interactive_fig.write_html(str(html_path), config=config_plotly)
                logger.info(f"Interactive UMAP figure saved to {html_path}")
            if show_figure in ["plotly", "all"]:
                interactive_fig.show(config=config_plotly)

    except Exception as e:
        logger.exception(str(e))
        raise e
    finally:
        logger_config.remove_file_handler()
