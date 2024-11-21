import numpy as np
import networkx as nx
import random
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple


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

    Parameters:
    -----------
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

    Returns:
    --------
    Dict[str, Tuple[float, float]]
        A dictionary mapping node names to their positions (x, y).
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
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


def merge_func(
    transition_matrix: np.ndarray,
    n_clusters: int,
    motif_norm: np.ndarray,
    merge_sel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge nodes in a graph based on a selection criterion.

    Parameters:
    -----------
    transition_matrix : np.ndarray
        The transition matrix of the graph.
    n_clusters : int
        The number of clusters.
    motif_norm : np.ndarray
        The normalized motif matrix.
    merge_sel : int
        The merge selection criterion.
        - 0: Merge nodes with highest transition probability.
        - 1: Merge nodes with lowest cost.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the merged nodes.
    """
    if merge_sel == 0:
        # merge nodes with highest transition probability
        cost = np.max(transition_matrix)
        merge_nodes = np.where(cost == transition_matrix)
    elif merge_sel == 1:
        cost_temp = 100
        for i in range(n_clusters):
            for j in range(n_clusters):
                try:
                    cost = motif_norm[i] + motif_norm[j] / np.abs(
                        transition_matrix[i, j] + transition_matrix[j, i]
                    )
                except ZeroDivisionError:
                    print(
                        "Error: Transition probabilities between motif "
                        + str(i)
                        + " and motif "
                        + str(j)
                        + " are zero."
                    )
                if cost <= cost_temp:
                    cost_temp = cost
                    merge_nodes = (np.array([i]), np.array([j]))
    else:
        raise ValueError("Invalid merge selection criterion. Please select 0 or 1.")
    return merge_nodes


def graph_to_tree(
    motif_usage: np.ndarray,
    transition_matrix: np.ndarray,
    n_clusters: int,
    merge_sel: int = 1,
) -> nx.Graph:
    """
    Convert a graph to a tree.

    Parameters:
    -----------
    motif_usage : np.ndarray
        The motif usage matrix.
    transition_matrix : np.ndarray
        The transition matrix of the graph.
    n_clusters : int
        The number of clusters.
    merge_sel : int, optional
        The merge selection criterion. Defaults to 1.
        - 0: Merge nodes with highest transition probability.
        - 1: Merge nodes with lowest cost.

    Returns:
    --------
    nx.Graph
        The tree.
    """
    if merge_sel == 1:
        # motif_usage_temp = np.load(path_to_file+'/behavior_quantification/motif_usage.npy')
        motif_usage_temp = motif_usage
        motif_usage_temp_colsum = motif_usage_temp.sum(axis=0)
        motif_norm = motif_usage_temp / motif_usage_temp_colsum
        motif_norm_temp = motif_norm.copy()
    else:
        motif_norm_temp = None

    merging_nodes = []
    hierarchy_nodes = []
    trans_mat_temp = transition_matrix.copy()
    is_leaf = np.ones((n_clusters), dtype="int")
    node_label = []
    leaf_idx = []

    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1) == 0)
        reduction = len(temp) + 1
    else:
        reduction = 1

    for i in range(n_clusters - reduction):
        nodes = merge_func(
            trans_mat_temp,
            n_clusters,
            motif_norm_temp,
            merge_sel,
        )

        if np.size(nodes) >= 2:
            nodes = np.array([nodes[0][0], nodes[1][0]])

        if is_leaf[nodes[0]] == 1:
            is_leaf[nodes[0]] = 0
            node_label.append("leaf_left_" + str(i))
            leaf_idx.append(1)

        elif is_leaf[nodes[0]] == 0:
            node_label.append("h_" + str(i) + "_" + str(nodes[0]))
            leaf_idx.append(0)

        if is_leaf[nodes[1]] == 1:
            is_leaf[nodes[1]] = 0
            node_label.append("leaf_right_" + str(i))
            hierarchy_nodes.append("h_" + str(i) + "_" + str(nodes[1]))
            leaf_idx.append(1)

        elif is_leaf[nodes[1]] == 0:
            node_label.append("h_" + str(i) + "_" + str(nodes[1]))
            hierarchy_nodes.append("h_" + str(i) + "_" + str(nodes[1]))
            leaf_idx.append(0)

        merging_nodes.append(nodes)

        node1_trans_x = trans_mat_temp[nodes[0], :]
        node2_trans_x = trans_mat_temp[nodes[1], :]

        node1_trans_y = trans_mat_temp[:, nodes[0]]
        node2_trans_y = trans_mat_temp[:, nodes[1]]

        new_node_trans_x = node1_trans_x + node2_trans_x
        new_node_trans_y = node1_trans_y + node2_trans_y

        trans_mat_temp[nodes[1], :] = new_node_trans_x
        trans_mat_temp[:, nodes[1]] = new_node_trans_y

        trans_mat_temp[nodes[0], :] = 0
        trans_mat_temp[:, nodes[0]] = 0

        trans_mat_temp[nodes[1], nodes[1]] = 0

        if merge_sel == 1:
            motif_norm_1 = motif_norm_temp[nodes[0]]
            motif_norm_2 = motif_norm_temp[nodes[1]]
            new_motif = motif_norm_1 + motif_norm_2

            motif_norm_temp[nodes[0]] = 0
            # motif_norm_temp[nodes[1]] = 0
            motif_norm_temp[nodes[1]] = new_motif

    merge = np.array(merging_nodes)

    T = nx.Graph()

    T.add_node("Root")
    node_dict = {}

    if leaf_idx[-1] == 0:
        temp_node = "h_" + str(merge[-1, 1]) + "_" + str(28)
        T.add_edge(temp_node, "Root")
        node_dict[merge[-1, 1]] = temp_node

    if leaf_idx[-1] == 1:
        T.add_edge(merge[-1, 1], "Root")

    if leaf_idx[-2] == 0:
        temp_node = "h_" + str(merge[-1, 0]) + "_" + str(28)
        T.add_edge(temp_node, "Root")
        node_dict[merge[-1, 0]] = temp_node

    if leaf_idx[-2] == 1:
        T.add_edge(merge[-1, 0], "Root")

    idx = len(leaf_idx) - 3

    if np.any(transition_matrix.sum(axis=1) == 0):
        temp = np.where(transition_matrix.sum(axis=1) == 0)
        reduction = len(temp) + 2
    else:
        reduction = 2

    for i in range(n_clusters - reduction)[::-1]:

        if leaf_idx[idx - 1] == 1:
            if merge[i, 1] in node_dict:
                T.add_edge(merge[i, 0], node_dict[merge[i, 1]])
            else:
                T.add_edge(merge[i, 0], temp_node)

        if leaf_idx[idx] == 1:
            if merge[i, 1] in node_dict:
                T.add_edge(merge[i, 1], node_dict[merge[i, 1]])
            else:
                T.add_edge(merge[i, 1], temp_node)

        if leaf_idx[idx] == 0:
            new_node = "h_" + str(merge[i, 1]) + "_" + str(i)
            if merge[i, 1] in node_dict:
                T.add_edge(node_dict[merge[i, 1]], new_node)
            else:
                T.add_edge(temp_node, new_node)

            if leaf_idx[idx - 1] == 1:
                temp_node = new_node
                node_dict[merge[i, 1]] = new_node
            else:
                new_node_2 = "h_" + str(merge[i, 0]) + "_" + str(i)
                T.add_edge(node_dict[merge[i, 1]], new_node_2)
                node_dict[merge[i, 1]] = new_node
                node_dict[merge[i, 0]] = new_node_2

        elif leaf_idx[idx - 1] == 0:
            new_node = "h_" + str(merge[i, 0]) + "_" + str(i)
            if merge[i, 1] in node_dict:
                T.add_edge(node_dict[merge[i, 1]], new_node)
            else:
                T.add_edge(temp_node, new_node)
            node_dict[merge[i, 0]] = new_node

            if leaf_idx[idx] == 1:
                temp_node = new_node
            else:
                new_node = "h_" + str(merge[i, 1]) + "_" + str(i)
                T.add_edge(temp_node, new_node)
                node_dict[merge[i, 1]] = new_node
                temp_node = new_node

        idx -= 2

    return T


def draw_tree(
    T: nx.Graph,
    fig_width: float = 200.0,
    usage_dict: Dict[str, float] = dict(),
) -> None:
    """
    Draw a tree.

    Parameters:
    -----------
    T : nx.Graph
        The tree to be drawn.
    fig_width : int, optional
        The width of the figure. Defaults to 10.

    Returns:
    --------
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
    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()


def _traverse_tree_cutline(
    T: nx.Graph,
    node: List[str],
    traverse_list: List[str],
    cutline: int,
    level: int,
    community_bag: List[List[str]],
    community_list: List[str] = None,
) -> List[List[str]]:
    """
    DEPRECATED in favor of bag_nodes_by_cutline.
    Helper function for tree traversal with a cutline.

    Parameters:
    -----------
    T : nx.Graph
        The tree to be traversed.
    node : List[str]
        Current node being traversed.
    traverse_list : List[str]
        List of traversed nodes.
    cutline : int
        The cutline level.
    level : int
        The current level in the tree.
    community_bag : List[List[str]]
        List of community bags.
    community_list : List[str], optional
        List of nodes in the current community bag.

    Returns:
    --------
    List[List[str]]
        List of lists community bags.
    """
    cmap = plt.get_cmap("tab10")
    traverse_list.append(node[0])
    if community_list is not None and type(node[0]) is not str:
        community_list.append(node[0])
    neighbors = list(T.neighbors(node[0]))

    # This gets intersection Nodes: Nodes that are not leaves
    if len(neighbors) == 3:
        for nei in neighbors:
            if nei in traverse_list:
                neighbors.remove(nei)

    if len(neighbors) > 1:
        if nx.shortest_path_length(T, "Root", node[0]) == cutline:
            # create new list
            traverse_list1 = []
            traverse_list2 = []
            community_bag = _traverse_tree_cutline(
                T,
                [neighbors[0]],
                traverse_list,
                cutline,
                level + 1,
                community_bag,
                traverse_list1,
            )
            community_bag = _traverse_tree_cutline(
                T,
                [neighbors[1]],
                traverse_list,
                cutline,
                level + 1,
                community_bag,
                traverse_list2,
            )
            joined_list = traverse_list1 + traverse_list2
            community_bag.append(joined_list)
            if type(node[0]) is not str:  # append itself
                community_bag.append([node[0]])
        else:
            community_bag = _traverse_tree_cutline(
                T,
                [neighbors[0]],
                traverse_list,
                cutline,
                level + 1,
                community_bag,
                community_list,
            )
            community_bag = _traverse_tree_cutline(
                T,
                [neighbors[1]],
                traverse_list,
                cutline,
                level + 1,
                community_bag,
                community_list,
            )

    return community_bag


def traverse_tree_cutline(
    T: nx.Graph,
    root_node: str | None = None,
    cutline: int = 2,
) -> List[List[str]]:
    """
    DEPRECATED in favor of bag_nodes_by_cutline.
    Traverse a tree with a cutline and return the community bags.

    Parameters:
    -----------
    T : nx.Graph
        The tree to be traversed.
    root_node : str, optional
        The root node of the tree. If None, traversal starts from the root.
    cutline : int, optional
        The cutline level.

    Returns:
    --------
    List[List[str]]
        List of community bags.
    """
    if root_node is None:
        node = ["Root"]
    else:
        node = [root_node]
    traverse_list = []
    color_map = []
    community_bag = []
    level = 0
    community_bag = _traverse_tree_cutline(
        T,
        node,
        traverse_list,
        cutline,
        level,
        color_map,
        community_bag,
    )
    return community_bag


# Added by Luiz: 2024-11-12
# This generalizes the problem of bagging nodes from the Tree using a cutline
def bag_nodes_by_cutline(
    tree: nx.Graph,
    cutline: int = 2,
    root: str = "Root",
):
    """
    Bag nodes of a tree by a cutline.

    Parameters:
    -----------
    tree : nx.Graph
        The tree to be bagged.
    cutline : int, optional
        The cutline level. Defaults to 2.
    root : str, optional
        The root node of the tree. Defaults to 'Root'.

    Returns:
    --------
    List[List[str]]
        List of bags of nodes.
    """
    if not tree.has_node(root):
        raise ValueError(f"Root node '{root}' not found in the tree.")
    if cutline < 0:
        raise ValueError("Cutline must be a non-negative integer.")

    directed_tree = nx.bfs_tree(tree, source=root)
    leaves = [n for n in directed_tree.nodes() if directed_tree.out_degree(n) == 0]
    bags = {}

    for leaf in leaves:
        path = nx.shortest_path(directed_tree, source=root, target=leaf)
        depth = len(path) - 1
        if depth >= cutline:
            ancestor_at_cutline = path[cutline]
        else:
            ancestor_at_cutline = leaf  # Each leaf in its own bag
        bags.setdefault(ancestor_at_cutline, []).append(leaf)

    return list(bags.values())
