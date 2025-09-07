import os
import json
import torch
from pathlib import Path
from vame.logging.logger import VameLogger
from vame.model.rnn_model import RNN_VAE


logger_config = VameLogger(__name__)
logger = logger_config.logger


def load_training_metadata(config: dict) -> dict:
    """
    Load training metadata to get keypoints used during training.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Training metadata containing keypoints_used and other parameters.

    Raises
    ------
    FileNotFoundError
        If metadata.json file is not found.
    ValueError
        If metadata is invalid or missing required fields.
    """
    project_path = Path(config["project_path"])
    metadata_path = project_path / "data" / "train" / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Training metadata not found at {metadata_path}. "
            "Please ensure you have run vame.create_trainset() before segmentation."
        )

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in metadata file: {e}")

    # Validate required fields
    if "parameters" not in metadata:
        raise ValueError("Metadata missing 'parameters' field")

    if "keypoints_used" not in metadata["parameters"]:
        raise ValueError("Metadata missing 'keypoints_used' in parameters")

    logger.info(f"Loaded training metadata with {len(metadata['parameters']['keypoints_used'])} keypoints")
    logger.info(f"Keypoints used during training: {metadata['parameters']['keypoints_used']}")

    return metadata


def load_model(config: dict, model_name: str, fixed: bool = True) -> RNN_VAE:
    """Load the VAME model.

    Args:
        config (dict): Configuration dictionary.
        model_name (str): Name of the model.
        fixed (bool): Fixed or variable length sequences.

    Returns
        RNN_VAE: Loaded VAME model.
    """
    # load Model
    ZDIMS = config["zdims"]
    FUTURE_DECODER = config["prediction_decoder"]
    TEMPORAL_WINDOW = config["time_window"] * 2
    FUTURE_STEPS = config["prediction_steps"]
    NUM_FEATURES = config["num_features"]
    hidden_size_layer_1 = config["hidden_size_layer_1"]
    hidden_size_layer_2 = config["hidden_size_layer_2"]
    hidden_size_rec = config["hidden_size_rec"]
    hidden_size_pred = config["hidden_size_pred"]
    dropout_encoder = config["dropout_encoder"]
    dropout_rec = config["dropout_rec"]
    dropout_pred = config["dropout_pred"]
    softplus = config["softplus"]

    logger.info("Loading model... ")

    model = RNN_VAE(
        TEMPORAL_WINDOW,
        ZDIMS,
        NUM_FEATURES,
        FUTURE_DECODER,
        FUTURE_STEPS,
        hidden_size_layer_1,
        hidden_size_layer_2,
        hidden_size_rec,
        hidden_size_pred,
        dropout_encoder,
        dropout_rec,
        dropout_pred,
        softplus,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    model.load_state_dict(
        torch.load(
            os.path.join(
                config["project_path"],
                "model",
                "best_model",
                model_name + "_" + config["project_name"] + ".pkl",
            )
        )
    )
    model.eval()

    return model
