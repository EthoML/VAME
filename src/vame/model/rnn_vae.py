import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

from vame.model.dataloader import SEQUENCE_DATASET
from vame.model.rnn_model import RNN_VAE
from vame.schemas.states import TrainModelFunctionSchema, save_state
from vame.logging.logger import VameLogger, TqdmToLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger
tqdm_to_logger = TqdmToLogger(logger)

# make sure torch uses cuda for GPU computing
use_gpu = torch.cuda.is_available()

if use_gpu:
    logger.info("GPU detected")
    logger.info(f"GPU used: {torch.cuda.get_device_name(0)}")
else:
    logger.info("No GPU found... proceeding with CPU (slow!)")
    torch.device("cpu")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruction_loss(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """
    Compute the reconstruction loss between input and reconstructed data.

    Parameters
    ----------
    x : torch.Tensor
        Input data tensor.
    x_tilde : torch.Tensor
        Reconstructed data tensor.
    reduction : str
        Type of reduction for the loss.

    Returns
    -------
    torch.Tensor
        Reconstruction loss.
    """
    mse_loss = nn.MSELoss(reduction=reduction) # Maybe concider sum over mean? but seems to be dependent on KL loss using mean
    rec_loss = mse_loss(x_tilde, x)
    return rec_loss


def future_reconstruction_loss(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """
    Compute the future reconstruction loss between input and predicted future data.

    Parameters
    ----------
    x : torch.Tensor
        Input future data tensor.
    x_tilde : torch.Tensor
        Reconstructed future data tensor.
    reduction : str
        Type of reduction for the loss.

    Returns
    -------
    torch.Tensor
        Future reconstruction loss.
    """
    mse_loss = nn.MSELoss(reduction=reduction) # Same as above if changed
    rec_loss = mse_loss(x_tilde, x)
    return rec_loss


def cluster_loss(
    H: torch.Tensor,
    kloss: int,
    lmbda: float,
    batch_size: int,
) -> torch.Tensor:
    """
    Compute the cluster loss.

    Parameters
    ----------
    H : torch.Tensor
        Latent representation tensor.
    kloss : int
        Number of clusters.
    lmbda : float
        Lambda value for the loss.
    batch_size : int
        Size of the batch.

    Returns
    -------
    torch.Tensor
        Cluster loss.
    """
    gram_matrix = (H.T @ H) / batch_size
    _, sv_2, _ = torch.svd(gram_matrix)
    sv = torch.sqrt(sv_2[:kloss])
    loss = torch.sum(sv)
    return lmbda * loss


def kullback_leibler_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Kullback-Leibler divergence loss.
    See Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 - https://arxiv.org/abs/1312.6114

    Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the latent distribution.
    logvar : torch.Tensor
        Log variance of the latent distribution.

    Returns
    -------
    torch.Tensor
        Kullback-Leibler divergence loss.
    """
    # I'm using torch.mean() here as the sum() version depends on the size of the latent vector
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def kl_annealing(
    epoch: int,
    KL_START: int,
    ANNEALTIME: int,
    function: str,
) -> float:
    """
    Anneal the Kullback-Leibler loss to let the model learn first the reconstruction of the data
    before the KL loss term gets introduced.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    KL_START : int
        Epoch number to start annealing the loss.
    ANNEALTIME : int
        Annealing time.
    function : str
        Annealing function type.

    Returns
    -------
    float
        Annealed weight value for the loss.
    """
    if epoch > KL_START:
        if function == "linear":
            new_weight = min(1, (epoch - KL_START) / (ANNEALTIME))

        elif function == "sigmoid":
            new_weight = float(1 / (1 + np.exp(-0.9 * (epoch - ANNEALTIME))))
        else:
            raise NotImplementedError('currently only "linear" and "sigmoid" are implemented')

        return new_weight
    else:
        new_weight = 0
        return new_weight


def gaussian(
    ins: torch.Tensor,
    is_training: bool,
    seq_len_half: int,
    std_n: float = 0.8,
) -> torch.Tensor:
    """
    Add Gaussian noise to the input data.

    Parameters
    ----------
    ins : torch.Tensor
        Input data tensor.
    is_training : bool
        Whether it is training mode.
    seq_len_half : int
        Length of the sequence. (Half of the timewindow)
    std_n : float
        Standard deviation for the Gaussian noise.

    Returns
    -------
    torch.Tensor
        Noisy input data tensor.
    """
    if is_training:
        emp_std = ins.std(1) * std_n
        emp_std = emp_std.unsqueeze(2).repeat(1, 1, seq_len_half)
        emp_std = emp_std.permute(0, 2, 1)
        noise = Variable(ins.data.new(ins.size()).normal_(0, 1))
        return ins + (noise * emp_std)
    return ins


def train(
    train_loader: Data.DataLoader,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ANNEAL_FUNCTION: str,
    GAMMA: float,
    BETA: float,
    KL_START: int,
    ANNEALTIME: int,
    TEMPORAL_WINDOW: int,
    FUTURE_DECODER: bool,
    FUTURE_STEPS: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    MSE_REC_REDUCTION: str,
    MSE_PRED_REDUCTION: str,
    kloss: int,
    klmbda: float,
    bsize: int,
    NOISE: bool,
) -> Tuple[float, float, float, float, float, float]:
    """
    Train the model.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    epoch : int
        Current epoch number.
    model : nn.Module
        Model to be trained.
    optimizer : Optimizer
        Optimizer for training.
    ANNEAL_FUNCTION : str
        Annealing function type.
    BETA : float
        Beta value for the loss.
    KL_START : int
        Epoch number to start annealing the loss.
    ANNEALTIME : int
        Annealing time.
    TEMPORAL_WINDOW : int
        Length of the sequence.
    FUTURE_DECODER : bool
        Whether a future decoder is used.
    FUTURE_STEPS : int
        Number of future steps to predict.
    scheduler : lr_scheduler._LRScheduler
        Learning rate scheduler.
    MSE_REC_REDUCTION : str
        Reduction type for MSE reconstruction loss.
    MSE_PRED_REDUCTION : str
        Reduction type for MSE prediction loss.
    kloss : int
        Number of clusters for cluster loss.
    klmbda : float
        Lambda value for cluster loss.
    bsize : int
        Size of the batch.
    NOISE : bool
        Whether to add Gaussian noise to the input.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        Kullback-Leibler weight, train loss, K-means loss, KL loss,
        MSE loss, future loss.
    """
    # toggle model to train mode
    model.train()
    train_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    fut_loss = 0.0
    # loss = 0.0
    seq_len_half = int(TEMPORAL_WINDOW / 2)
    num_batches = len(train_loader)

    for data_item in train_loader:
        data_item = Variable(data_item)
        data_item = data_item.permute(0, 2, 1)

        data = data_item[:, :seq_len_half, :].type("torch.FloatTensor").to(DEVICE)
        fut = data_item[:, seq_len_half : seq_len_half + FUTURE_STEPS, :].type("torch.FloatTensor").to(DEVICE)
        
        if NOISE:
            data_gaussian = gaussian(data, True, seq_len_half)
        else:
            data_gaussian = data
        
        if FUTURE_DECODER:
            data_tilde, future, latent, mu, logvar = model(data_gaussian)
            fut_rec_loss = future_reconstruction_loss(fut, future, MSE_PRED_REDUCTION)
            fut_loss += fut_rec_loss.item()
        else:
            data_tilde, latent, mu, logvar = model(data_gaussian)

        rec_loss = reconstruction_loss(data, data_tilde, MSE_REC_REDUCTION)
        kl_loss = kullback_leibler_loss(mu, logvar)
        kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
        kl_weight = kl_annealing(epoch, KL_START, ANNEALTIME, ANNEAL_FUNCTION)
        loss = rec_loss + (GAMMA * fut_rec_loss) + (BETA * kl_loss) + kl_weight * kmeans_loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        mse_loss += rec_loss.item()
        kullback_loss += kl_loss.item()
        kmeans_losses += kmeans_loss.item()
        # total_loss += loss

    # be sure scheduler is called before optimizer in >1.1 pytorch
    scheduler.step(train_loss)

    avg_train_loss = train_loss / num_batches
    avg_mse_loss = mse_loss / num_batches
    avg_kullback_loss = kullback_loss / num_batches
    avg_kmeans_losses = kmeans_losses / num_batches
    avg_fut_loss = 0.0

    if FUTURE_DECODER:
        avg_fut_loss = fut_loss / num_batches

    logger.info(
        "Train loss: {:.3f}, MSE-Loss: {:.3f}, MSE-Future-Loss {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}".format(
            avg_train_loss,
            avg_mse_loss,
            GAMMA * avg_fut_loss,
            BETA * kl_weight * avg_kullback_loss,
            kl_weight * avg_kmeans_losses,
            kl_weight,
        )
    )

    return (
        kl_weight, # weight
        avg_train_loss, # train loss
        kl_weight * avg_kmeans_losses, # km_loss
        avg_kullback_loss, # kl_loss
        avg_mse_loss, # mse_tain_loss
        GAMMA * avg_fut_loss, # fut_loss
    )


def test(
    test_loader: Data.DataLoader,
    model: nn.Module,
    GAMMA: float,
    BETA: float,
    kl_weight: float,
    TEMPORAL_WINDOW: int,
    MSE_REC_REDUCTION: str,
    MSE_PRED_REDUCTION: str,
    kloss: str,
    klmbda: float,
    FUTURE_DECODER: bool,
    FUTURE_STEPS: int,
    bsize: int,
) -> Tuple[float, float, float]:
    """
    Evaluate the model on the test dataset.

    Parameters
    ----------
    test_loader : DataLoader
        DataLoader for the test dataset.
    model : nn.Module
        The trained model.
    BETA : float
        Beta value for the VAE loss.
    kl_weight : float
        Weighting factor for the KL divergence loss.
    TEMPORAL_WINDOW : int
        Length of the sequence.
    MSE_REC_REDUCTION : str
        Reduction method for the MSE loss.
    MSE_PRED_REDUCTION : str
        Reduction type for MSE prediction loss.
    kloss : str
        Loss function for K-means clustering.
    klmbda : float
        Lambda value for K-means loss.
    FUTURE_DECODER : bool
        Flag indicating whether to use a future decoder.
    FUTURE_STEPS : int
        Number of future steps to predict.
    bsize :int
        Batch size.

    Returns
    -------
    Tuple[float, float, float]
        Tuple containing MSE loss per item, total test loss per item,
        and K-means loss weighted by the kl_weight.
    """
    # toggle model to inference mode
    model.eval()
    test_loss = 0.0
    mse_loss = 0.0
    kullback_loss = 0.0
    kmeans_losses = 0.0
    fut_loss = 0.0
    seq_len_half = int(TEMPORAL_WINDOW / 2)
    num_batches = len(test_loader)

    with torch.no_grad():
        for data_item in test_loader:
            # we're only going to infer, so no autograd at all required
            data_item = Variable(data_item)
            data_item = data_item.permute(0, 2, 1)
            data = data_item[:, :seq_len_half, :].type("torch.FloatTensor").to(DEVICE)
            fut = data_item[:, seq_len_half : seq_len_half + FUTURE_STEPS, :].type("torch.FloatTensor").to(DEVICE)

            if FUTURE_DECODER:
                recon_images, future, latent, mu, logvar = model(data)
                fut_rec_loss = future_reconstruction_loss(fut, future, MSE_PRED_REDUCTION)
                fut_loss += fut_rec_loss.item()
            else:
                recon_images, latent, mu, logvar = model(data)

            rec_loss = reconstruction_loss(data, recon_images, MSE_REC_REDUCTION)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            loss = rec_loss + (GAMMA * fut_rec_loss) + (BETA * kl_loss) + (kl_weight * kmeans_loss) # explicit clustering focus
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)
            
            test_loss += loss.item()
            mse_loss += rec_loss.item()
            kullback_loss += kl_loss.item()
            kmeans_losses += kmeans_loss.item()
    
    avg_test_loss = test_loss / num_batches
    avg_mse_loss = mse_loss / num_batches
    avg_kullback_loss = kullback_loss / num_batches
    avg_kmeans_losses = kmeans_losses / num_batches
    avg_fut_loss = 0.0

    if FUTURE_DECODER:
        avg_fut_loss = fut_loss / num_batches
    logger.info(
        "Test loss: {:.3f}, MSE-Loss: {:.3f}, MSE-Future-Loss: {:.3f} KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}".format(
            avg_test_loss,
            avg_mse_loss,
            GAMMA * avg_fut_loss,
            BETA * kl_weight * avg_kullback_loss, # order of operations error..?
            kl_weight * avg_kmeans_losses, # order of operations error..?
        )
    )
    return (
        avg_test_loss, # current_loss
        avg_mse_loss # mse_test_loss
        # avg_test_loss # test_loss
        # avg_fut_loss # fut_loss
        )


@save_state(model=TrainModelFunctionSchema)
def train_model(
    config: dict,
    save_logs: bool = False,
) -> None:
    """
    Train Variational Autoencoder using the configuration file values.
    Fills in the values in the "train_model" key of the states.json file.
    Creates files at:
    - project_name/
        - model/
            - best_model/
                - snapshots/
                    - model_name_Project_epoch_0.pkl
                    - ...
                - model_name_Project.pkl
            - model_losses/
                - fut_losses_VAME.npy
                - kl_losses_VAME.npy
                - kmeans_losses_VAME.npy
                - mse_test_losses_VAME.npy
                - mse_train_losses_VAME.npy
                - test_losses_VAME.npy
                - train_losses_VAME.npy
                - weight_values_VAME.npy
            - pretrained_model/


    Parameters
    ----------
    config : dict
        Configuration dictionary.
    save_logs : bool, optional
        Whether to save the logs, by default False.

    Returns
    -------
    None
    """
    config = config
    try:
        tqdm_logger_stream = None
        if save_logs:
            tqdm_logger_stream = TqdmToLogger(logger)
            log_path = Path(config["project_path"]) / "logs" / "train_model.log"
            logger_config.add_file_handler(str(log_path))

        model_name = config["model_name"]
        pretrained_weights = config["pretrained_weights"]
        pretrained_model = config["pretrained_model"]
        fixed = config["egocentric_data"]

        logger.info("Train Variational Autoencoder - model name: %s \n" % model_name )
        if not os.path.exists(os.path.join(config["project_path"], "model", config['testing_name'], "best_model")):
            os.makedirs(os.path.join(config["project_path"], "model", config['testing_name'], "best_model"))
            os.makedirs(os.path.join(config["project_path"], "model", config['testing_name'], "best_model", "snapshots"))
            os.makedirs(os.path.join(config["project_path"], "model", config['testing_name'], "model_losses"))
            if config['testing_name'] != '':
                os.makedirs(os.path.join(config["project_path"], "model", config['testing_name'], "evaluate"))

        # make sure torch uses cuda for GPU computing
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            logger.info("GPU active: {}".format(torch.cuda.is_available()))
            logger.info("GPU used: {}".format(torch.cuda.get_device_name(0)))
        else:
            torch.device("cpu")
            logger.info("warning, a GPU was not found... proceeding with CPU (slow!) \n")
            # raise NotImplementedError('GPU Computing is required!')

        # HYPERPARAMETERS
        # General
        SEED = 19
        TRAIN_BATCH_SIZE = config["batch_size"]
        TEST_BATCH_SIZE = int(config["batch_size"] / 4)
        EPOCHS = config["max_epochs"]
        ZDIMS = config["zdims"]
        GAMMA = config['gamma']
        BETA = config["beta"]
        SNAPSHOT = config["model_snapshot"]
        LEARNING_RATE = config["learning_rate"]
        NUM_FEATURES = config["num_features"]
        if not fixed:
            NUM_FEATURES = NUM_FEATURES - 3
        TEMPORAL_WINDOW = config["time_window"] * 2
        FUTURE_DECODER = config["prediction_decoder"]
        FUTURE_STEPS = config["prediction_steps"]

        # RNN
        HIDDEN_SIZE_LAYER_1 = config["hidden_size_layer_1"]
        HIDDEN_SIZE_LAYER_2 = config["hidden_size_layer_2"]
        HIDDEN_SIZE_REC = config["hidden_size_rec"]
        HIDDEN_SIZE_PRED = config["hidden_size_pred"]
        DROPOUT_ENCODER = config["dropout_encoder"]
        DROPOUT_REC = config["dropout_rec"]
        DROPOUT_PRED = config["dropout_pred"]
        NOISE = config["noise"]
        SOFTPLUS = config["softplus"]

        # Loss
        MSE_REC_REDUCTION = config["mse_reconstruction_reduction"]
        MSE_PRED_REDUCTION = config["mse_prediction_reduction"]
        KMEANS_LOSS = config["kmeans_loss"]
        KMEANS_LAMBDA = config["kmeans_lambda"]
        KL_START = config["kl_start"]
        ANNEALTIME = config["annealtime"]
        ANNEAL_FUNCTION = config["anneal_function"]
        OPTIMIZER_SCHEDULER = config["scheduler"]
        SCHEDULER_STEP_SIZE = config["scheduler_step_size"]

        BEST_LOSS = 999999
        convergence = 0
        logger.info(
            "Latent Dimensions: %d, Time window: %d, Batch Size: %d, Beta: %.4f, Gamma: %d, lr: %.4f\n"
            % (ZDIMS, config["time_window"], TRAIN_BATCH_SIZE, BETA, GAMMA, LEARNING_RATE)
        )

        # simple logging of diverse losses
        train_losses = []
        test_losses = []
        kmeans_losses = []
        kl_losses = []
        weight_values = []
        mse_train_losses = []
        mse_test_losses = []
        total_losses = []
        fut_losses = []

        torch.manual_seed(SEED)
        RNN = RNN_VAE
        torch.cuda.manual_seed(SEED)
        model = RNN(
                TEMPORAL_WINDOW,
                ZDIMS,
                NUM_FEATURES,
                FUTURE_DECODER,
                FUTURE_STEPS,
                HIDDEN_SIZE_LAYER_1,
                HIDDEN_SIZE_LAYER_2,
                HIDDEN_SIZE_REC,
                HIDDEN_SIZE_PRED,
                DROPOUT_ENCODER,
                DROPOUT_REC,
                DROPOUT_PRED,
                SOFTPLUS,
            ).to(DEVICE)

        if pretrained_weights:
            try:
                logger.info(
                    "Loading pretrained weights from model: %s\n"
                    % os.path.join(
                        config["project_path"],
                        "model",
                        config['testing_name'],
                        "best_model",
                        pretrained_model + "_" + config["project_name"] + ".pkl",
                    )
                )
                model.load_state_dict(
                    torch.load(
                        os.path.join(
                            config["project_path"],
                            "model",
                            config['testing_name'],
                            "best_model",
                            pretrained_model + "_" + config["project_name"] + ".pkl",
                        )
                    )
                )
                KL_START = 0
                ANNEALTIME = 1
            except Exception:
                logger.info(
                    "No file found at %s\n"
                    % os.path.join(
                        config["project_path"],
                        "model",
                        config['testing_name'],
                        "best_model",
                        pretrained_model + "_" + config["project_name"] + ".pkl",
                    )
                )
                try:
                    logger.info("Loading pretrained weights from %s\n" % pretrained_model)
                    model.load_state_dict(torch.load(pretrained_model))
                    KL_START = 0
                    ANNEALTIME = 1
                except Exception:
                    logger.error("Could not load pretrained model. Check file path in config.yaml.")

        """ DATASET """
        trainset = SEQUENCE_DATASET(
            os.path.join(config["project_path"], "data", "train", ""),
            data="train_seq.npy",
            train=True,
            temporal_window=TEMPORAL_WINDOW,
        )
        testset = SEQUENCE_DATASET(
            os.path.join(config["project_path"], "data", "train", ""),
            data="test_seq.npy",
            train=False,
            temporal_window=TEMPORAL_WINDOW,
        )

        train_loader = Data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

        if OPTIMIZER_SCHEDULER:
            logger.info(
                "Scheduler step size: %d, Scheduler gamma: %.2f\n" % (SCHEDULER_STEP_SIZE, config["scheduler_gamma"])
            )
            # Thanks to @alexcwsmith for the optimized scheduler contribution
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=config["scheduler_gamma"],
                patience=config["scheduler_step_size"],
                threshold=1e-3,
                threshold_mode="rel",
                verbose=True,
            )
        else:
            scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=1, last_epoch=-1)

        best_model_save_file = os.path.join(
                            config["project_path"],
                            "model",
                            config['testing_name'],
                            "best_model",
                            model_name + "_" + config["project_name"] + ".pkl",)
        logger.info(f"Start training... 11 kdm_dev")
        for epoch in tqdm(
            range(1, EPOCHS),
            desc="Training Model",
            unit="epoch",
            file=tqdm_logger_stream,
        ):
            
            weight, train_loss, km_loss, kl_loss, mse_train_loss, fut_loss = train(
                train_loader,
                epoch,
                model,
                optimizer,
                ANNEAL_FUNCTION,
                GAMMA,
                BETA,
                KL_START,
                ANNEALTIME,
                TEMPORAL_WINDOW,
                FUTURE_DECODER,
                FUTURE_STEPS,
                scheduler,
                MSE_REC_REDUCTION,
                MSE_PRED_REDUCTION,
                KMEANS_LOSS,
                KMEANS_LAMBDA,
                TRAIN_BATCH_SIZE,
                NOISE,
            )
            current_loss, mse_test_loss = test(
                test_loader,
                model,
                GAMMA,
                BETA,
                weight,
                TEMPORAL_WINDOW,
                MSE_REC_REDUCTION,
                MSE_PRED_REDUCTION,
                KMEANS_LOSS,
                KMEANS_LAMBDA,
                FUTURE_DECODER,
                FUTURE_STEPS,
                TEST_BATCH_SIZE,
            )

            # logging losses
            train_losses.append(train_loss)
            test_losses.append(current_loss)
            kmeans_losses.append(km_loss)
            kl_losses.append(kl_loss)
            weight_values.append(weight)
            mse_train_losses.append(mse_train_loss)
            mse_test_losses.append(mse_test_loss)
            fut_losses.append(fut_loss)

            # save best model
            if weight > 0.99 and current_loss <= BEST_LOSS and epoch > ANNEALTIME + 10:
                BEST_LOSS = current_loss
                logger.info("Saving model!")
                torch.save(model.state_dict(), best_model_save_file)
                convergence = 0
            else:
                if weight > 0.99 and epoch > ANNEALTIME + 10:
                    convergence += 1

            if epoch % SNAPSHOT == 0:
                logger.info("Saving model snapshot! did we get here?\n")
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        config["project_path"],
                        "model",
                        config['testing_name'],
                        "best_model",
                        "snapshots",
                        model_name + "_" + config["project_name"] + "_epoch_" + str(epoch) + ".pkl",
                    ),
                )

            if convergence > config["model_convergence"]:
                if not os.path.isfile(best_model_save_file):
                    logger.info("Saving model!")
                    torch.save(model.state_dict(), best_model_save_file)


                logger.info("Finished training...")
                logger.info(
                    "Model converged. Please check your model with vame.evaluate_model(). \n"
                    "You can also re-run vame.trainmodel() to further improve your model. \n"
                    'Make sure to set _pretrained_weights_ in your config.yaml to "true" \n'
                    "and plug your current model name into _pretrained_model_. \n"
                    'Hint: Set "model_convergence" in your config.yaml to a higher value. \n'
                    "\n"
                    "Next: \n"
                    "Use vame.segment_session() to identify behavioral motifs in your dataset!"
                )
                break

            # save logged losses
            loss_dict = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'kmeans_losses' : kmeans_losses,
                'kl_losses' : kl_losses,
                'weight_values' : weight_values,
                'mse_train_losses' : mse_train_losses,
                'mse_test_losses' : mse_test_losses,
                'fut_losses' : fut_losses
            }

            for name, loss in loss_dict.items():
                np.save(
                os.path.join(
                    config["project_path"],
                    "model",
                    config['testing_name'],
                    "model_losses",
                    f"{name}_{model_name}",
                    ),
                    loss
                )
                
            logger.info("\n")

        if convergence < config["model_convergence"]:
            logger.info("Finished training...")
            logger.info(
                "Model seems to have not reached convergence. You may want to check your model \n"
                "with vame.evaluate_model(). If your satisfied you can continue. \n"
                "Use vame.segment_session() to identify behavioral motifs! \n"
                "OPTIONAL: You can re-run vame.train_model() to improve performance."
            )

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise e
    finally:
        logger_config.remove_file_handler()
