import random
from typing import Dict, Tuple, Literal
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt


def hierarchy_pos(
    G: nx.Graph,
    root: str | None = None,
    width: float = 0.5,
    vert_gap: float = 0.2,
    vert_loc: float = 0,
    xcenter: float = 0.5,
) -> Dict[str, Tuple[float, float]]:
    """
    Positions nodes in a tree-like layout.
    Ref: From Joel's answer at https://stackoverflow.com/a/29597209/2966723.

    Parameters
    ----------
    G : nx.Graph
        The input graph. Must be a tree.
    root : str, optional
        The root node of the tree. If None, the function selects a root node based on graph type.
        Defaults to None.
    width : float, optional
        The horizontal space assigned to each level. Defaults to 0.5.
    vert_gap : float, optional
        The vertical gap between levels. Defaults to 0.2.
    vert_loc : float, optional
        The vertical location of the root node. Defaults to 0.
    xcenter : float, optional
        The horizontal location of the root node. Defaults to 0.5.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        A dictionary mapping node names to their positions (x, y).
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G,
        root,
        width=1.0,
        vert_gap=0.2,
        vert_loc=0,
        xcenter=0.5,
        pos=None,
        parent=None,
    ):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def draw_tree(
    T: nx.Graph,
    fig_width: float = 20.0,
    usage_dict: Dict[str, float] = dict(),
    save_to_file: bool = True,
    show_figure: bool = False,
    results_dir: str | None = None,
) -> None:
    """
    Draw a tree.

    Parameters
    ----------
    T : nx.Graph
        The tree to be drawn.
    fig_width : int, optional
        The width of the figure. Defaults to 20.
    usage_dict : Dict[str, float], optional
        Dictionary mapping node names to their usage values. Defaults to empty dictionary.
    save_to_file : bool, optional
        Flag indicating whether to save the plot. Defaults to True.
    show_figure : bool, optional
        Flag indicating whether to show the plot. Defaults to False.
    results_dir : str, optional
        The directory to save the plot. Defaults to None.

    Returns
    -------
    None
    """
    # pos = nx.drawing.layout.fruchterman_reingold_layout(T)
    pos = hierarchy_pos(
        G=T,
        root="Root",
        width=10.0,
        vert_gap=0.1,
        vert_loc=0,
        xcenter=50,
    )
    # Nodes appearances
    # Nodes sizes are mapped to a scale between 100 and 61prin00, depending on the usage of the node
    node_labels = dict()
    node_sizes = []
    node_colors = []
    for k in list(T.nodes):
        if isinstance(k, str):
            node_labels[k] = ""
            node_sizes.append(50)
            node_colors.append("#000000")
        else:
            node_labels[k] = str(k)
            size = usage_dict.get(str(k), 0.5)
            node_sizes.append(100 + size * 6000)
            node_colors.append("#46a7e8")

    fig_width = min(max(fig_width, 10.0), 30.0)
    fig = plt.figure(
        num=2,
        figsize=(fig_width, 20.0),
    )
    nx.draw_networkx(
        G=T,
        pos=pos,
        with_labels=True,
        labels=node_labels,
        node_size=node_sizes,
        node_color=node_colors,
    )
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    if save_to_file and results_dir:
        save_fig_path = Path(results_dir) / "tree.png"
        save_fig_pdf_path = Path(results_dir) / "tree.pdf"
        plt.savefig(save_fig_path, bbox_inches="tight")
        plt.savefig(save_fig_pdf_path, bbox_inches="tight")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def visualize_hierarchical_tree(
    config: dict,
    segmentation_algorithm: Literal["hmm", "kmeans"],
) -> None:
    """
    Visualizes the hierarchical tree.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    segmentation_algorithm : Literal["hmm", "kmeans"]
        Segmentation algorithm.

    Returns
    -------
    None
    """
    n_clusters = config["n_clusters"]
    fig_path = (
        Path(config["project_path"])
        / "results"
        / "community_cohort"
        / f"{segmentation_algorithm}-{n_clusters}"
        / "tree.png"
    )
    if not fig_path.exists():
        raise FileNotFoundError(f"Tree figure not found at {fig_path}.")
    img = plt.imread(fig_path)
    plt.figure(figsize=(n_clusters, n_clusters))
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()