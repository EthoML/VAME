import os
import torch
from pathlib import Path
import torch.utils.data as Data
from typing import Optional

from vame.schemas.states import EvaluateModelFunctionSchema, save_state
from vame.model.rnn_vae import RNN_VAE
from vame.model.dataloader import SEQUENCE_DATASET
from vame.logging.logger import VameLogger
from vame.visualization.model import plot_reconstruction


logger_config = VameLogger(__name__)
logger = logger_config.logger

use_gpu = torch.cuda.is_available()
if use_gpu:
    pass
else:
    torch.device("cpu")


def eval_temporal(
    config: dict,
    use_gpu: bool,
    model_name: str,
    fixed: bool,
    snapshot: Optional[str] = None,
    suffix: Optional[str] = None,
) -> None:
    """
    Evaluate the temporal aspects of the trained model.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    use_gpu : bool
        Flag indicating whether to use GPU for evaluation.
    model_name : str
        Name of the model.
    fixed : bool
        Flag indicating whether the data is fixed or not.
    snapshot : str, optional
        Path to the model snapshot. Defaults to None.
    suffix : str, optional
        Suffix for the saved plot filename. Defaults to None.

    Returns
    -------
    None
    """
    SEED = 19
    ZDIMS = config["zdims"]
    FUTURE_DECODER = config["prediction_decoder"]
    TEMPORAL_WINDOW = config["time_window"] * 2
    FUTURE_STEPS = config["prediction_steps"]
    NUM_FEATURES = config["num_features"]
    if not fixed:
        NUM_FEATURES = NUM_FEATURES - 3
    TEST_BATCH_SIZE = 64
    hidden_size_layer_1 = config["hidden_size_layer_1"]
    hidden_size_layer_2 = config["hidden_size_layer_2"]
    hidden_size_rec = config["hidden_size_rec"]
    hidden_size_pred = config["hidden_size_pred"]
    dropout_encoder = config["dropout_encoder"]
    dropout_rec = config["dropout_rec"]
    dropout_pred = config["dropout_pred"]
    softplus = config["softplus"]

    filepath = os.path.join(config["project_path"], "model")

    seq_len_half = int(TEMPORAL_WINDOW / 2)
    if use_gpu:
        torch.cuda.manual_seed(SEED)
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
        ).cuda()
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
    else:
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
        ).to()
        if not snapshot:
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        config["project_path"],
                        "model",
                        "best_model",
                        model_name + "_" + config["project_name"] + ".pkl",
                    ),
                    map_location=torch.device("cpu"),
                )
            )
        elif snapshot:
            model.load_state_dict(torch.load(snapshot), map_location=torch.device("cpu"))
    model.eval()  # toggle evaluation mode

    testset = SEQUENCE_DATASET(
        os.path.join(config["project_path"], "data", "train", ""),
        data="test_seq.npy",
        train=False,
        temporal_window=TEMPORAL_WINDOW,
        logger_config=logger_config,
    )
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

    if not snapshot:
        plot_reconstruction(
            filepath,
            test_loader,
            seq_len_half,
            model,
            model_name,
            FUTURE_DECODER,
            FUTURE_STEPS,
        )  # , suffix=suffix
    elif snapshot:
        plot_reconstruction(
            filepath,
            test_loader,
            seq_len_half,
            model,
            model_name,
            FUTURE_DECODER,
            FUTURE_STEPS,
            suffix=suffix,
        )  # ,
    # if use_gpu:
    #     plot_loss(
    #         config=config,
    #         filepath=filepath,
    #         model_name=model_name,
    #         show_figure=False,
    #     )
    # else:
    #     plot_loss(
    #         config=config,
    #         filepath=filepath,
    #         model_name=model_name,
    #         show_figure=False,
    #     )
    #     # pass #note, loading of losses needs to be adapted for CPU use #TODO


@save_state(model=EvaluateModelFunctionSchema)
def evaluate_model(
    config: dict,
    use_snapshots: bool = False,
    save_logs: bool = True,
) -> None:
    """
    Evaluate the trained model.
    Fills in the values in the "evaluate_model" key of the states.json file.
    Saves the evaluation results to:
    - project_name/
        - model/
            - evaluate/

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    use_snapshots : bool, optional
        Whether to plot for all snapshots or only the best model. Defaults to False.
    save_logs : bool, optional
        Whether to save logs. Defaults to True.

    Returns
    -------
    None
    """
    try:
        project_path = Path(config["project_path"]).resolve()
        if save_logs:
            log_path = project_path / "logs" / "evaluate_model.log"
            logger_config.add_file_handler(str(log_path))

        model_name = config["model_name"]
        fixed = config["egocentric_data"]

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("Using CUDA")
            logger.info("GPU active: {}".format(torch.cuda.is_available()))
            logger.info("GPU used: {}".format(torch.cuda.get_device_name(0)))
        else:
            torch.device("cpu")
            logger.info("CUDA is not working, or a GPU is not found; using CPU!")

        logger.info(f"Evaluation of model: {model_name}")
        if not use_snapshots:
            eval_temporal(
                config=config,
                use_gpu=use_gpu,
                model_name=model_name,
                fixed=fixed,
            )
        elif use_snapshots:
            snapshots = os.listdir(os.path.join(str(project_path), "model", "best_model", "snapshots"))
            for snap in snapshots:
                fullpath = os.path.join(str(project_path), "model", "best_model", "snapshots", snap)
                epoch = snap.split("_")[-1]
                eval_temporal(
                    config=config,
                    use_gpu=use_gpu,
                    model_name=model_name,
                    fixed=fixed,
                    snapshot=fullpath,
                    suffix="snapshot" + str(epoch),
                )

        logger.info(
            f"You can find the results of the evaluation in '{project_path}/model/evaluate/' \n"
            "OPTIONS:\n"
            "- vame.segment_session() to identify behavioral motifs.\n"
            "- re-run the model for further fine tuning. Check again with vame.evaluate_model()"
        )
    except Exception as e:
        logger.exception(f"An error occurred during model evaluation: {e}")
    finally:
        logger_config.remove_file_handler()
