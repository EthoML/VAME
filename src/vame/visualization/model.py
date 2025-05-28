import os
from pathlib import Path
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt


def plot_loss(
    config: dict,
    model_name: Optional[str] = None,
    save_to_file: bool = False,
    show_figure: bool = True,
) -> None:
    """
    Plot the losses of the trained model.
    Saves the plot to:
    - project_name/
        - model/
            - evaluate/
                - mse_and_kl_loss_model_name.png

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    model_name : str
        Name of the model. Defaults to None, in which case the model name in config is used.
    save_to_file : bool, optional
        Flag indicating whether to save the plot. Defaults to False.
    show_figure : bool, optional
        Flag indicating whether to show the plot. Defaults to True.

    Returns
    -------
    None
    """
    if model_name is None:
        model_name = config["model_name"]
    basepath = os.path.join(config["project_path"], "model", "model_losses")
    train_loss = np.load(os.path.join(basepath, "train_losses_" + model_name + ".npy"))
    test_loss = np.load(os.path.join(basepath, "test_losses_" + model_name + ".npy"))
    mse_loss_train = np.load(os.path.join(basepath, "mse_train_losses_" + model_name + ".npy"))
    mse_loss_test = np.load(os.path.join(basepath, "mse_test_losses_" + model_name + ".npy"))
    km_losses = np.load(os.path.join(basepath, "kmeans_losses_" + model_name + ".npy"))
    kl_loss = np.load(os.path.join(basepath, "kl_losses_" + model_name + ".npy"))
    fut_loss = np.load(os.path.join(basepath, "fut_losses_" + model_name + ".npy"))

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle(f"Losses of model: {model_name}")
    ax1.set(xlabel="Epochs", ylabel="loss [log-scale]")
    ax1.set_yscale("log")
    ax1.plot(train_loss, label="Train-Loss")
    ax1.plot(test_loss, label="Test-Loss")
    ax1.plot(mse_loss_train, label="MSE-Train-Loss")
    ax1.plot(mse_loss_test, label="MSE-Test-Loss")
    ax1.plot(km_losses, label="KMeans-Loss")
    ax1.plot(kl_loss, label="KL-Loss")
    ax1.plot(fut_loss, label="Prediction-Loss")
    ax1.legend()

    if save_to_file:
        evaluate_path = os.path.join(config["project_path"], "model", "evaluate")
        fig.savefig(os.path.join(evaluate_path, "mse_and_kl_loss_" + model_name + ".png"))

    if show_figure:
        plt.show()
    else:
        plt.close(fig)
