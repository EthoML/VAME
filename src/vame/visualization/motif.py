from typing import Literal
from pathlib import Path
import matplotlib.pyplot as plt


def visualize_motif_tree(
    config: dict,
    segmentation_algorithm: Literal["hmm", "kmeans"],
) -> None:
    """
    Visualizes the motif tree.

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
