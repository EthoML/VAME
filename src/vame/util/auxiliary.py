from importlib.metadata import version
import os
import json
import yaml
import ruamel.yaml
from pathlib import Path
from typing import Tuple, Any
from enum import Enum

from vame.schemas.states import save_state, UpdateConfigFunctionSchema


def get_version() -> str:
    """
    Gets the VAME package version from pyproject.toml.

    Returns
    -------
    str
        The version string.
    """
    return version("vame-py")


def _convert_enums_to_values(obj: Any) -> Any:
    """
    Recursively converts enum values to their string representations.

    Parameters
    ----------
    obj : Any
        The object to convert.

    Returns
    -------
    Any
        The converted object with enum values replaced by their string representations.
    """
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {key: _convert_enums_to_values(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_enums_to_values(item) for item in obj]
    return obj


def create_config_template() -> Tuple[dict, ruamel.yaml.YAML]:
    """
    Creates a template for the config.yaml file.

    Returns
    -------
    Tuple[dict, ruamel.yaml.YAML]
        A tuple containing the template dictionary and the Ruamel YAML instance.
    """
    yaml_str = """\
# Project configurations
    vame_version:
    project_name:
    project_path:
    creation_datetime:
    session_names:
    \n
# Data
    all_data:
    keypoints:
    \n
# Preprocessing:
    egocentric_data:
    pose_confidence:
    robust:
    iqr_factor:
    savgol_filter:
    savgol_length:
    savgol_order:
    test_fraction:
    \n
# RNN model general hyperparameter:
    model_name:
    pretrained_model:
    pretrained_weights:
    num_features:
    batch_size:
    max_epochs:
    model_snapshot:
    model_convergence:
    transition_function:
    beta:
    beta_norm:
    zdims:
    learning_rate:
    time_window:
    prediction_decoder:
    prediction_steps:
    noise:
    scheduler:
    scheduler_step_size:
    scheduler_gamma:
    #Note the optimal scheduler threshold below can vary greatly (from .1-.0001) between experiments.
    #You are encouraged to read the torch.optim.ReduceLROnPlateau docs to understand the threshold to use.
    scheduler_threshold:
    softplus:
    \n
# Segmentation:
    n_clusters:
    segmentation_algorithms:
    hmm_trained: False
    load_data:
    individual_segmentation:
    random_state_kmeans:
    n_init_kmeans:
    \n
# Video writer:
    length_of_motif_video:
    \n
# UMAP parameter:
    min_dist:
    n_neighbors:
    random_state:
    num_points:
    \n
#--------------------------------------------------------
# ONLY CHANGE ANYTHING BELOW IF YOU ARE FAMILIAR WITH RNN MODELS
# RNN encoder hyperparamter:
    hidden_size_layer_1:
    hidden_size_layer_2:
    dropout_encoder:
    \n
# RNN reconstruction hyperparameter:
    hidden_size_rec:
    dropout_rec:
    n_layers:
    \n
# RNN prediction hyperparamter:
    hidden_size_pred:
    dropout_pred:
    \n
# RNN loss hyperparameter:
    mse_reconstruction_reduction:
    mse_prediction_reduction:
    kmeans_loss:
    kmeans_lambda:
    anneal_function:
    kl_start:
    annealtime:
"""
    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return (cfg_file, ruamelFile)


def read_config(config_file: str) -> dict:
    """
    Reads structured config file defining a project.

    Parameters
    ----------
    config_file : str
        Path to the config file.

    Returns
    -------
    dict
        The contents of the config file as a dictionary.
    """
    ruamelFile = ruamel.yaml.YAML()
    path = Path(config_file)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                config = ruamelFile.load(f)
                curr_dir = os.path.dirname(config_file)
                if config["project_path"] != curr_dir:
                    config["project_path"] = curr_dir
                    write_config(
                        config_path=config_file,
                        config=config,
                    )
        except Exception as err:
            if len(err.args) > 2:
                if err.args[2] == "could not determine a constructor for the tag '!!python/tuple'":
                    with open(path, "r") as ymlfile:
                        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
                        write_config(
                            config_path=config_file,
                            config=config,
                        )
                else:
                    raise
    else:
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or that you passed the path of the config file correctly!"
        )
    return config


def write_config(
    config_path: str,
    config: dict,
) -> None:
    """
    Write structured config file.

    Parameters
    ----------
    config_path : str
        Path to the config file.
    config : dict
        Dictionary containing the config data.
    """
    with open(config_path, "w") as cf:
        cfg_file, ruamelFile = create_config_template()

        # Convert any enum values to strings before writing
        config = _convert_enums_to_values(config)

        for key in config.keys():
            cfg_file[key] = config[key]
        ruamelFile.dump(cfg_file, cf)


@save_state(model=UpdateConfigFunctionSchema)
def update_config(
    config: dict,
    config_update: dict,
) -> dict:
    config_path = Path(config["project_path"]) / "config.yaml"
    config.update(config_update)
    write_config(config_path, config)
    return config


def read_states(config: dict) -> dict:
    """
    Reads the states.json file.

    Parameters
    ----------
    config : dict
        Dictionary containing the config data.

    Returns
    -------
    dict
        The contents of the states.json file as a dictionary.
    """
    states_path = Path(config["project_path"]) / "states" / "states.json"
    with open(states_path, "r") as f:
        states = json.load(f)
    return states
