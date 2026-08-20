import os
import tqdm
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from joblib import Parallel, delayed

from vame.schemas.states import save_state, SegmentSessionFunctionSchema
from vame.logging.logger import VameLogger, TqdmToLogger
from vame.model.rnn_model import RNN_VAE
from vame.io.load_poses import read_pose_estimation_file
from vame.util.cli import get_sessions_from_user_input
from vame.util.model_util import load_model, load_training_metadata
from vame.util.auxiliary import get_device
from vame.preprocessing.extra import validate_extra_features
from vame.preprocessing.to_model import format_xarray_for_rnn


logger_config = VameLogger(__name__)
logger = logger_config.logger


def embed_latent_vectors(
    config: dict,
    sessions: List[str],
    fixed: bool,
    read_from_variable: str = "position_processed",
    overwrite: bool = False,
    tqdm_stream: Union[TqdmToLogger, None] = None,
) -> List[np.ndarray]:
    """
    Embed latent vectors for the given sessions using the VAME model.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    sessions : List[str]
        List of session names.
    fixed : bool
        Whether the model is fixed.
    read_from_variable : str, optional
        Variable to read from the dataset. Defaults to "position_processed".
    overwrite : bool, optional
        Whether to overwrite existing latent vector files. Defaults to False.
    tqdm_stream : TqdmToLogger, optional
        TQDM Stream to redirect the tqdm output to logger.

    Returns
    -------
    List[np.ndarray]
        List of latent vectors for all sessions.
    """
    project_path = config["project_path"]
    model_name = config["model_name"]
    temp_win = config["time_window"]
    num_features = config["num_features"]
    model = None

    logger.info("---------------------------------------------------------------------")
    logger.info(f"Embedding latent vectors for {model_name} model")

    # Load training metadata to get keypoints used during training
    try:
        training_metadata = load_training_metadata(config)
        keypoints_used = training_metadata["parameters"]["keypoints_used"]
        extra_features_used = list(training_metadata["parameters"].get("extra_features") or [])
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load training metadata: {e}")
        logger.warning("Using all available keypoints - this may cause shape mismatch errors")
        keypoints_used = None
        extra_features_used = []

    # Validate extra features against the *training* metadata (not the live
    # config), so inference fails fast if the model's inputs aren't available.
    validate_extra_features(
        config=config,
        sessions=sessions,
        extra_features=extra_features_used,
    )

    latent_vector_sessions = []
    for session in sessions:
        latent_vector_path = Path(project_path) / "results" / session / config["model_name"] / "latent_vectors.npy"
        if latent_vector_path.exists():
            if not overwrite:
                logger.info(f"Latent vector for {session} already exists, skipping...")
                latent_vector = np.load(latent_vector_path)
                latent_vector_sessions.append(latent_vector)
                continue
            else:
                logger.info(f"Latent vector for {session} already exists, but will be overwritten.")
        logger.info(f"Embedding of latent vector for file {session}")

        # Load the model, if not yet loaded
        if model is None:
            device = get_device()
            model = load_model(config, model_name, fixed)

        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        # Format the data for the RNN model using the same keypoints as training
        data, _ = format_xarray_for_rnn(
            ds=ds,
            read_from_variable=read_from_variable,
            keypoints=keypoints_used,
            extra_features=extra_features_used,
        )

        latent_vector_list = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(data.shape[1] - temp_win + 1), file=tqdm_stream):
                data_sample_np = data[:, i : temp_win + i].T
                data_sample_np = np.reshape(data_sample_np, (1, temp_win, num_features))
                h_n = model.encoder(torch.from_numpy(data_sample_np).float().to(device))
                mu, _, _ = model.lmbda(h_n)
                latent_vector_list.append(mu.cpu().data.numpy())

        latent_vector = np.concatenate(latent_vector_list, axis=0)

        # Save latent vector to file
        np.save(latent_vector_path, latent_vector)

        latent_vector_sessions.append(latent_vector)

    return latent_vector_sessions


def embed_latent_vectors_optimized(
    config: dict,
    sessions: List[str],
    fixed: bool,
    read_from_variable: str = "position_processed",
    overwrite: bool = False,
    batch_size: int = 64,
    tqdm_stream: Union[TqdmToLogger, None] = None,
) -> List[np.ndarray]:
    """
    Optimized version of embed_latent_vectors with batch processing and vectorized operations.

    This function provides significant performance improvements over the original implementation:
    - Vectorized sliding window creation (no data copying)
    - Batch processing of multiple windows simultaneously
    - GPU memory optimization with pre-allocated tensors
    - Model optimizations for faster inference

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    sessions : List[str]
        List of session names.
    fixed : bool
        Whether the model is fixed.
    read_from_variable : str, optional
        Variable to read from the dataset. Defaults to "position_processed".
    overwrite : bool, optional
        Whether to overwrite existing latent vector files. Defaults to False.
    batch_size : int, optional
        Number of windows to process simultaneously. Defaults to 64.
        Larger values use more GPU memory but may be faster.
    tqdm_stream : TqdmToLogger, optional
        TQDM Stream to redirect the tqdm output to logger.

    Returns
    -------
    List[np.ndarray]
        List of latent vectors for all sessions.
    """
    project_path = config["project_path"]
    model_name = config["model_name"]
    temp_win = config["time_window"]
    num_features = config["num_features"]
    model = None

    logger.info("---------------------------------------------------------------------")
    logger.info(f"Embedding latent vectors for {model_name} model (OPTIMIZED)")

    # Load training metadata to get keypoints used during training
    try:
        training_metadata = load_training_metadata(config)
        keypoints_used = training_metadata["parameters"]["keypoints_used"]
        extra_features_used = list(training_metadata["parameters"].get("extra_features") or [])
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load training metadata: {e}")
        logger.warning("Using all available keypoints - this may cause shape mismatch errors")
        keypoints_used = None
        extra_features_used = []

    # Validate extra features against the *training* metadata (not the live
    # config), so inference fails fast if the model's inputs aren't available.
    validate_extra_features(
        config=config,
        sessions=sessions,
        extra_features=extra_features_used,
    )

    latent_vector_sessions = []

    for session in sessions:
        latent_vector_path = Path(project_path) / "results" / session / config["model_name"] / "latent_vectors.npy"
        if latent_vector_path.exists():
            if not overwrite:
                logger.info(f"Latent vector for {session} already exists, skipping...")
                latent_vector = np.load(latent_vector_path)
                latent_vector_sessions.append(latent_vector)
                continue
            else:
                logger.info(f"Latent vector for {session} already exists, but will be overwritten.")

        logger.info(f"Embedding of latent vector for file {session}")

        # Load the model, if not yet loaded
        if model is None:
            device = get_device()
            model = load_model(config, model_name, fixed)

            # Model optimizations
            model.eval()  # Ensure model is in evaluation mode
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        # Format the data for the RNN model using the same keypoints as training
        data, _ = format_xarray_for_rnn(
            ds=ds,
            read_from_variable=read_from_variable,
            keypoints=keypoints_used,
            extra_features=extra_features_used,
        )

        # Calculate number of windows
        n_windows = data.shape[1] - temp_win + 1
        if n_windows <= 0:
            logger.warning(f"Session {session} has insufficient data for time window {temp_win}")
            latent_vector_sessions.append(np.array([]))
            continue

        # Create all sliding windows at once using vectorized operations
        logger.info(f"Creating {n_windows} sliding windows for session {session}")

        # Use stride_tricks for efficient sliding window creation (no data copying)
        try:
            # Transpose data to (time, features) for sliding window
            data_transposed = data.T  # Shape: (time, features)

            # Use sliding_window_view on each axis separately
            windows = np.lib.stride_tricks.sliding_window_view(
                data_transposed,
                window_shape=temp_win,
                axis=0
            )
            # Result shape: (n_windows, temp_win, num_features)

            # Verify shape is correct
            if windows.shape != (n_windows, temp_win, num_features):
                raise ValueError(f"Unexpected window shape: {windows.shape}")

        except Exception as e:
            logger.warning(f"Stride tricks failed ({e}), using fallback method")
            windows = np.zeros((n_windows, temp_win, num_features))
            for i in range(n_windows):
                windows[i] = data[:, i:i + temp_win].T

        # Pre-allocate output array
        latent_dim = config["zdims"]
        latent_vectors = np.zeros((n_windows, latent_dim), dtype=np.float32)

        # Process windows in batches
        n_batches = (n_windows + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in tqdm.tqdm(range(n_batches),
                                     desc=f"Processing {session}",
                                     file=tqdm_stream):
                # Calculate batch boundaries
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_windows)
                current_batch_size = end_idx - start_idx

                # Get batch of windows
                batch_windows = windows[start_idx:end_idx]

                # Convert to tensor
                batch_tensor = torch.from_numpy(batch_windows).type("torch.FloatTensor")

                batch_tensor = batch_tensor.to(device)

                try:
                    # Process entire batch through encoder
                    h_n = model.encoder(batch_tensor)
                    mu, _, _ = model.lmbda(h_n)

                    # Store results
                    latent_vectors[start_idx:end_idx] = mu.cpu().numpy()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Reduce batch size and retry
                        logger.warning(f"GPU out of memory, reducing batch size from {batch_size}")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif device.type == "mps":
                            torch.mps.empty_cache()

                        # Process windows one by one for this batch
                        for i in range(current_batch_size):
                            single_window = batch_windows[i:i+1]
                            single_tensor = torch.from_numpy(single_window).type("torch.FloatTensor")
                            single_tensor = single_tensor.to(device)

                            h_n = model.encoder(single_tensor)
                            mu, _, _ = model.lmbda(h_n)
                            latent_vectors[start_idx + i] = mu.cpu().numpy()
                    else:
                        raise e

                # Clear GPU cache periodically
                if device.type == "cuda" and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                elif device.type == "mps" and batch_idx % 10 == 0:
                    torch.mps.empty_cache()

        # Save latent vector to file
        np.save(latent_vector_path, latent_vectors)
        latent_vector_sessions.append(latent_vectors)

        logger.info(f"Successfully processed {n_windows} windows for session {session}")

    return latent_vector_sessions


def get_motif_usage(
    session_labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Count motif usage from session label array.

    Parameters
    ----------
    session_labels : np.ndarray
        Array of session labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Array of motif usage counts.
    """
    motif_usage = np.zeros(n_clusters)
    for i in range(n_clusters):
        motif_count = np.sum(session_labels == i)
        motif_usage[i] = motif_count
    # Include warning if any unused motifs are present
    unused_motifs = np.where(motif_usage == 0)[0]
    if unused_motifs.size > 0:
        logger.info(f"Warning: The following motifs are unused: {unused_motifs}")
    return motif_usage


def save_session_data(
    project_path: str,
    session: str,
    model_name: str,
    label: np.ndarray,
    cluster_centers: np.ndarray,
    motif_usage: np.ndarray,
    n_clusters: int,
    segmentation_algorithm: str,
):
    """
    Saves pose segmentation data for given session.

    Parameters
    ----------
    project_path: str
        Path to the vame project folder.
    session: str
        Session name.
    model_name: str
        Name of model
    label: np.ndarray
        Array of the session's motif labels.
    cluster_centers: np.ndarray
        Array of the session's kmeans cluster centers location in the latent space.
    motif_usage: np.ndarray
        Array of the session's motif usage counts.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm: str
        Type of segmentation method: 'kmeans', 'hmm', or 'hmm_warmstart'.

    Returns
    -------
    None
    """
    session_results_path = os.path.join(
        str(project_path),
        "results",
        session,
        model_name,
        segmentation_algorithm + "-" + str(n_clusters),
    )
    if not os.path.exists(session_results_path):
        try:
            os.mkdir(session_results_path)
        except OSError as error:
            logger.error(error)

    np.save(
        os.path.join(session_results_path, str(n_clusters) + "_" + segmentation_algorithm + "_label_" + session),
        label,
    )
    if segmentation_algorithm == "kmeans":
        np.save(
            os.path.join(session_results_path, "cluster_center_" + session),
            cluster_centers,
        )
    np.save(
        os.path.join(session_results_path, "motif_usage_" + session),
        motif_usage,
    )
    logger.info(f"Saved {session} segmentation data")


def _session_estep(model: "hmm.GaussianHMM", session_data: np.ndarray) -> Tuple[dict, float]:
    """
    The expensive, session-independent half of one Baum-Welch iteration:
    forward-backward + sufficient-statistics accumulation for a single
    session, against the model's current parameters. Designed to be run in
    parallel across sessions via joblib - safe to do so because sufficient
    statistics are purely additive across independent sequences (this is the
    same assumption hmmlearn's own sequential per-sequence loop relies on).
    """
    impl = {"scaling": model._fit_scaling, "log": model._fit_log}[model.implementation]
    lattice, logprob, posteriors, fwdlattice, bwdlattice = impl(session_data)
    stats = model._initialize_sufficient_statistics()
    model._accumulate_sufficient_statistics(stats, session_data, lattice, posteriors, fwdlattice, bwdlattice)
    return stats, logprob


def _merge_stats(stats_list: List[dict]) -> dict:
    """
    Sum per-session sufficient statistics into one combined stats dict.
    Values are either plain int (nobs) or np.ndarray; `+` handles both and
    always returns a new object, so no explicit .copy() is needed.
    """
    merged: dict = {}
    for stats in stats_list:
        for key, val in stats.items():
            merged[key] = val if key not in merged else merged[key] + val
    return merged


def parallel_gaussian_hmm_fit(
    model: "hmm.GaussianHMM",
    session_arrays: List[np.ndarray],
    n_jobs: int = 1,
) -> "hmm.GaussianHMM":
    """
    Equivalent to model.fit(np.concatenate(session_arrays), lengths=[...]),
    with the E-step (forward-backward per session) parallelized across
    processes instead of hmmlearn's own single-threaded per-sequence loop.
    Mirrors hmmlearn.base.BaseHMM.fit() exactly (same _init/_check/monitor_
    calls) - only how the E-step's stats/logprob get computed differs.

    Parameters
    ----------
    model : hmm.GaussianHMM
        A configured (not yet fit) GaussianHMM instance.
    session_arrays : List[np.ndarray]
        Per-session latent vector arrays.
    n_jobs : int, optional
        Passed to joblib.Parallel. 1 = sequential (no process overhead),
        -1 = use all available cores. Defaults to 1.

    Returns
    -------
    hmm.GaussianHMM
        The same model instance, fit in place.
    """
    X = check_array(np.concatenate(session_arrays, axis=0))
    lengths = np.asarray([len(s) for s in session_arrays])

    model._init(X, lengths)
    model._check()
    model.monitor_._reset()

    # backend="threading", not joblib's default "loky": hmmlearn's Cython core
    # releases the GIL during its numeric loops, so threads give real parallel
    # speedup. Process-based "loky" was measured ~260x slower here because each
    # worker process re-imports the full torch/CUDA-heavy module chain.
    with Parallel(n_jobs=n_jobs, backend="threading") as parallel:
        for _ in range(model.n_iter):
            results = parallel(delayed(_session_estep)(model, s) for s in session_arrays)
            stats_list, logprobs = zip(*results)
            stats = _merge_stats(stats_list)
            curr_logprob = sum(logprobs)

            lower_bound = model._compute_lower_bound(curr_logprob)
            model._do_mstep(stats)
            model.monitor_.report(lower_bound)
            if model.monitor_.converged:
                break

            if (model.transmat_.sum(axis=1) == 0).any():
                logger.warning(
                    "Some rows of transmat_ have zero sum because no "
                    "transition from the state was ever observed."
                )
    return model


def same_segmentation(
    config: dict,
    sessions: List[str],
    latent_vectors: List[np.ndarray],
    n_clusters: int,
    segmentation_algorithm: str,
) -> None:
    """
    Apply the same segmentation to all animals.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    sessions : List[str]
        List of session names.
    latent_vectors : List[np.ndarray]
        List of latent vector arrays.
    n_clusters : int
        Number of clusters.
    segmentation_algorithm : str
        Segmentation algorithm.

    Returns
    -------
    None
    """
    random_state = config.get("project_random_state", 42)

    # List of arrays containing each session's motif labels #[SRM, 10/28/24], recommend rename this and similar variables to allsessions_labels
    cluster_centers = []  # List of arrays containing each session's cluster centers
    motif_usages = []  # List of arrays containing each session's motif usages

    latent_vector_cat = np.concatenate(latent_vectors, axis=0)
    if segmentation_algorithm == "kmeans":
        logger.info("Using kmeans as segmentation algorithm!")
        kmeans = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=20,
        ).fit(latent_vector_cat)
        cluster_centers = kmeans.cluster_centers_
        # 1D, vector of all labels for the entire cohort
        labels = kmeans.predict(latent_vector_cat)

    elif segmentation_algorithm == "hmm":
        if not config["hmm_trained"]:
            logger.info("Using a HMM as segmentation algorithm!")
            hmm_n_iter = config.get("hmm_n_iter", 100)
            hmm_model = hmm.GaussianHMM(
                n_components=n_clusters,
                covariance_type="full",
                n_iter=hmm_n_iter,
                random_state=random_state,
                verbose=True,
            )
            hmm_model.fit(latent_vector_cat)
            labels = hmm_model.predict(latent_vector_cat)
            model_path = Path(config["project_path"]) / "results" / "hmm_trained.pkl"
            with open(model_path, "wb") as file:
                pickle.dump(hmm_model, file)
        else:
            logger.info("Using a pretrained HMM as segmentation algorithm!")
            model_path = Path(config["project_path"]) / "results" / "hmm_trained.pkl"
            with open(model_path, "rb") as file:
                hmm_model = pickle.load(file)
            labels = hmm_model.predict(latent_vector_cat)

    elif segmentation_algorithm == "hmm_warmstart":
        logger.info("Using a warm-started HMM as segmentation algorithm!")
        hmm_stride = config.get("hmm_warmstart_stride", 5)
        hmm_n_iter = config.get("hmm_warmstart_n_iter", 500)
        hmm_tol = config.get("hmm_warmstart_tol", 1e-2)
        hmm_n_jobs = config.get("hmm_warmstart_n_jobs", 1)

        n_features_dim = latent_vector_cat.shape[1]
        pretrained_path = config.get("hmm_warmstart_pretrained_path")
        warmstart_model = None

        if pretrained_path:
            logger.info(f"Warm-start stage 1: loading pretrained model from {pretrained_path}")
            with open(pretrained_path, "rb") as file:
                candidate = pickle.load(file)
            if candidate.n_components != n_clusters or candidate.n_features != n_features_dim:
                logger.warning(
                    f"Pretrained model at {pretrained_path} has n_components={candidate.n_components}, "
                    f"n_features={candidate.n_features}, but this project needs n_clusters={n_clusters}, "
                    f"zdims={n_features_dim}. Ignoring it and fitting stage 1 from scratch instead."
                )
            else:
                warmstart_model = candidate

        if warmstart_model is None:
            # A full-covariance Gaussian needs roughly n_features samples per state
            # just to be non-singular. Guard against strides that would leave too
            # little data for stage 1 on small/pilot datasets (falls back to no
            # striding rather than risk a singular-covariance fit).
            min_frames_needed = n_clusters * n_features_dim * 2
            strided_sessions = [v[::hmm_stride] for v in latent_vectors]
            total_strided_frames = sum(v.shape[0] for v in strided_sessions)
            if total_strided_frames < min_frames_needed:
                logger.warning(
                    f"hmm_warmstart_stride={hmm_stride} would leave only {total_strided_frames} frames "
                    f"for {n_clusters} clusters x {n_features_dim} dims, too few for a stable "
                    "full-covariance fit. Falling back to no striding for the warm-start pre-fit."
                )
                strided_sessions = latent_vectors
                total_strided_frames = sum(v.shape[0] for v in strided_sessions)

            # Stage 1: cheap fit on strided (subsampled) data to find a good rough
            # state structure. A cold (randomly initialized) full-resolution fit on
            # this kind of data has been observed to converge to a degenerate local
            # optimum with several near-zero self-transition ("flicker") states.
            # Pre-fitting on strided data and using its parameters to initialize the
            # full-resolution fit reliably avoids that failure mode.
            logger.info(
                f"Warm-start stage 1: fitting on {total_strided_frames} frames "
                f"across {len(strided_sessions)} sessions (stride={hmm_stride}, n_jobs={hmm_n_jobs})"
            )
            warmstart_model = hmm.GaussianHMM(
                n_components=n_clusters,
                covariance_type="full",
                n_iter=hmm_n_iter,
                tol=hmm_tol,
                random_state=random_state,
                verbose=True,
            )
            parallel_gaussian_hmm_fit(warmstart_model, strided_sessions, n_jobs=hmm_n_jobs)

        # Stage 2: fine-tune at full resolution, seeded from stage 1's (freshly
        # fit or pretrained) parameters. The pretrained path is never used as-is,
        # this step always re-fits it on the current project's own data.
        # Small/edge-case datasets can leave stage 1 with a degenerate covariance
        # (fails hmmlearn's validation the moment it's assigned below) - fall back
        # to a plain cold-started fit rather than crash the whole pipeline on it.
        hmm_model = None
        try:
            logger.info(f"Warm-start stage 2: fine-tuning on full-resolution data (n_jobs={hmm_n_jobs})")
            candidate_model = hmm.GaussianHMM(
                n_components=n_clusters,
                covariance_type="full",
                n_iter=hmm_n_iter,
                tol=hmm_tol,
                random_state=random_state,
                init_params="",  # skip auto-init, use the seeded params below
                verbose=True,
            )
            candidate_model.startprob_ = warmstart_model.startprob_
            candidate_model.transmat_ = warmstart_model.transmat_
            candidate_model.means_ = warmstart_model.means_
            candidate_model.covars_ = warmstart_model.covars_
            candidate_model.n_features = warmstart_model.n_features
            parallel_gaussian_hmm_fit(candidate_model, latent_vectors, n_jobs=hmm_n_jobs)
            hmm_model = candidate_model
        except ValueError as e:
            logger.warning(
                f"Warm-start seeding produced an unusable model ({e}); "
                "falling back to a cold-started HMM fit on the full-resolution data instead."
            )

        if hmm_model is None:
            hmm_model = hmm.GaussianHMM(
                n_components=n_clusters,
                covariance_type="full",
                n_iter=hmm_n_iter,
                tol=hmm_tol,
                random_state=random_state,
                verbose=True,
            )
            parallel_gaussian_hmm_fit(hmm_model, latent_vectors, n_jobs=hmm_n_jobs)

        labels = hmm_model.predict(latent_vector_cat)

        # Saved under its own filename so it can be reused as a future project's
        # hmm_warmstart_pretrained_path. Unlike "hmm"'s hmm_trained flag, this is
        # never auto-reused within *this* project on a later rerun - only an
        # explicit path passed by the caller triggers reuse, and stage 2 always
        # fine-tunes on the target data rather than using it as-is. That silent
        # same-project auto-reuse pattern is what caused the original "hmm"
        # pipeline to keep serving a stale, undertrained model across reruns;
        # this method intentionally doesn't repeat it.
        model_path = Path(config["project_path"]) / "results" / "hmm_warmstart_trained.pkl"
        with open(model_path, "wb") as file:
            pickle.dump(hmm_model, file)

    idx = 0  # start index for each session
    for i, session in enumerate(sessions):
        session_len = latent_vectors[i].shape[0]  # stop index of the session
        session_labels = labels[idx : idx + session_len]

        # Session's motif usage
        motif_usage = get_motif_usage(session_labels, n_clusters)
        motif_usages.append(motif_usage)
        idx += session_len  # updating the session start index
        save_session_data(
            project_path=config["project_path"],
            session=session,
            model_name=config["model_name"],
            label=session_labels,
            cluster_centers=cluster_centers,
            motif_usage=motif_usage,
            n_clusters=n_clusters,
            segmentation_algorithm=segmentation_algorithm,
        )


def individual_segmentation(
    config: dict,
    sessions: List[str],
    latent_vectors: List[np.ndarray],
    n_clusters: int,
) -> Tuple:
    """
    Apply individual segmentation to each session.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    sessions : List[str]
        List of session names.
    latent_vectors : List[np.ndarray]
        List of latent vector arrays.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    Tuple
        Tuple of labels, cluster centers, and motif usages.
    """
    random_state = config.get("project_random_state", 42)
    n_init = config["n_init_kmeans"]
    labels = []
    cluster_centers = []
    motif_usages = []
    for i, session in enumerate(sessions):
        logger.info(f"Processing session: {session}")
        kmeans = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        ).fit(latent_vectors[i])
        clust_center = kmeans.cluster_centers_
        label = kmeans.predict(latent_vectors[i])
        motif_usage = get_motif_usage(
            session_labels=label,
            n_clusters=n_clusters,
        )
        motif_usages.append(motif_usage)
        labels.append(label)
        cluster_centers.append(clust_center)

        save_session_data(
            config["project_path"],
            session,
            config["model_name"],
            labels[i],
            cluster_centers[i],
            latent_vectors[i],
            motif_usages[i],
            n_clusters,
            "kmeans",
        )
    return labels, cluster_centers, motif_usages


@save_state(model=SegmentSessionFunctionSchema)
def segment_session(
    config: dict,
    overwrite_segmentation: bool = False,
    overwrite_embeddings: bool = False,
    save_logs: bool = True,
    optimized: bool = True,
) -> None:
    """
    Perform pose segmentation using the VAME model.
    Fills in the values in the "segment_session" key of the states.json file.
    Creates files at:
    - project_name/
        - results/
            - hmm_trained.pkl
            - hmm_warmstart_trained.pkl
            - session/
                - model_name/
                    - latent_vectors.npy
                    - hmm-n_clusters/
                        - motif_usage_session.npy
                        - n_cluster_label_session.npy
                    - hmm_warmstart-n_clusters/
                        - motif_usage_session.npy
                        - n_cluster_label_session.npy
                    - kmeans-n_clusters/
                        - motif_usage_session.npy
                        - n_cluster_label_session.npy
                        - cluster_center_session.npy

    latent_vectors.npy contains the projection of the data into the latent space,
    for each frame of the video. Dimmentions: (n_frames, n_latent_features)

    motif_usage_session.npy contains the number of times each motif was used in the video.
    Dimmentions: (n_motifs,)

    n_cluster_label_session.npy contains the label of the cluster assigned to each frame.
    Dimmentions: (n_frames,)

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    overwrite_segmentation : bool, optional
        Whether to overwrite existing segmentation results. Defaults to False.
    overwrite_embeddings : bool, optional
        If True, runs embedding function and re-creates embeddings files, even if they already exist.
        Defaults to False.
    optimized : bool, optional
        If True, uses the optimized version of the embedding function.
        If False, uses the original version. Defaults to True.
    save_logs : bool, optional
        Whether to save logs. Defaults to True.

    Returns
    -------
    None
    """
    project_path = Path(config["project_path"]).resolve()
    # Reproducibility: seed all RNGs from project_random_state.
    from vame.util.seed import seed_everything

    seed_everything(config.get("project_random_state", 42))
    try:
        tqdm_stream = None
        if save_logs:
            log_path = project_path / "logs" / "pose_segmentation.log"
            logger_config.add_file_handler(str(log_path))
            tqdm_stream = TqdmToLogger(logger)

        model_name = config["model_name"]
        n_clusters = config["n_clusters"]
        fixed = config["egocentric_data"]
        segmentation_algorithms = config["segmentation_algorithms"]
        ind_seg = config["individual_segmentation"]

        # Get sessions to analyze
        sessions = []
        if config["all_data"] in ["Yes", "yes", "True", "true", True]:
            sessions = config["session_names"]
        else:
            sessions = get_sessions_from_user_input(
                config=config,
                action_message="run segmentation",
            )

        # Check if each session general results path exists
        for session in sessions:
            session_results_path = os.path.join(
                str(project_path),
                "results",
                session,
                model_name,
            )
            if not os.path.exists(session_results_path):
                os.mkdir(session_results_path)

        # Create latent vector files
        if optimized:
            latent_vectors = embed_latent_vectors_optimized(
                config=config,
                sessions=sessions,
                fixed=fixed,
                overwrite=overwrite_embeddings,
                tqdm_stream=tqdm_stream,
            )
        else:
            latent_vectors = embed_latent_vectors(
            config=config,
            sessions=sessions,
            fixed=fixed,
            overwrite=overwrite_embeddings,
            tqdm_stream=tqdm_stream,
        )

        logger.info("---------------------------------------------------------------------")
        logger.info("Pose segmentation for VAME model: %s \n" % model_name)
        for seg in segmentation_algorithms:
            # Checks if segment session was already processed before
            seg_results_path = os.path.join(
                str(project_path),
                "results",
                sessions[0],
                model_name,
                seg + "-" + str(n_clusters),
            )
            if os.path.exists(seg_results_path):
                if not overwrite_segmentation:
                    logger.info(
                        f"Segmentation for {seg} algorithm and cluster size {n_clusters} already exists, skipping..."
                    )
                    continue
                logger.info(
                    f"Segmentation for {seg} algorithm and cluster size {n_clusters} already exists, but will be overwritten."
                )
            logger.info(f"Starting segmentation for {seg} algorithm and cluster size {n_clusters}...")

            # Apply same or indiv segmentation of latent vectors for each session
            if ind_seg:
                logger.info(f"Apply individual segmentation of latent vectors for each session, {n_clusters} clusters")
                labels, cluster_center, motif_usages = individual_segmentation(
                    config=config,
                    sessions=sessions,
                    latent_vectors=latent_vectors,
                    n_clusters=n_clusters,
                )
            else:
                logger.info(f"Apply the same segmentation of latent vectors for all sessions, {n_clusters} clusters")
                same_segmentation(
                    config=config,
                    sessions=sessions,
                    latent_vectors=latent_vectors,
                    n_clusters=n_clusters,
                    segmentation_algorithm=seg,
                )

            logger.info(
                "You succesfully extracted motifs with VAME! From here, you can proceed running vame.community() "
                "to get the full picture of the spatiotemporal dynamic. To get an idea of the behavior captured by VAME, "
                "run vame.motif_videos(). This will leave you with short snippets of certain movements."
            )
            logger.info("To visualize the average motif usage across sessions, run vame.visualization.plot_motif_thresholding.")

    except Exception as e:
        logger.exception(f"An error occurred during pose segmentation: {e}")
    finally:
        logger_config.remove_file_handler()
