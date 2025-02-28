import os
import torch
from vame.logging.logger import VameLogger
from vame.model.rnn_model import RNN_VAE


logger_config = VameLogger(__name__)
logger = logger_config.logger


def load_model(config: dict, model_name: str, fixed: bool = True, testing_name = '') -> RNN_VAE:
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

    if not fixed:
        NUM_FEATURES = NUM_FEATURES - 3
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
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEVICE)

    model.load_state_dict(
        torch.load(
            os.path.join(
                config["project_path"],
                "model",
                testing_name,
                "best_model",
                model_name + "_" + config["project_name"] + ".pkl",
            )
        )
    )
    model.eval()

    return model
