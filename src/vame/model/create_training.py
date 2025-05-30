import os
import numpy as np
from pathlib import Path
from typing import List, Literal

from vame.logging.logger import VameLogger
from vame.schemas.states import CreateTrainsetFunctionSchema, save_state
from vame.io.load_poses import read_pose_estimation_file
from vame.preprocessing.to_model import format_xarray_for_rnn


logger_config = VameLogger(__name__)
logger = logger_config.logger


def traindata_aligned(
    config: dict,
    sessions: List[str] | None = None,
    test_fraction: float = 0.1,
    read_from_variable: str = "position_processed",
    split_mode: Literal["mode_1", "mode_2"] = "mode_2",
) -> None:
    """
    Create training dataset for aligned data.
    Save numpy arrays with the test/train info to the project folder.

    Parameters
    ----------
    config : dict
        Configuration parameters dictionary.
    sessions : List[str], optional
        List of session names. If None, all sessions will be used. Defaults to None.
    test_fraction : float, optional
        Fraction of data to use as test data. Defaults to 0.1.
    read_from_variable : str, optional
        Variable name to read from the processed data. Defaults to "position_processed".
    split_mode : Literal["mode_1", "mode_2"], optional
        Mode for splitting data into train/test sets:
        - mode_1: Original mode that takes the initial test_fraction portion of the combined data
                 for testing and the rest for training.
        - mode_2: Takes random continuous chunks from each session proportional to test_fraction
                 for testing and uses the remaining parts for training.
        Defaults to "mode_2".

    Returns
    -------
    None
    """
    project_path = config["project_path"]
    if sessions is None:
        sessions = config["session_names"]
    if test_fraction is None:
        test_fraction = config["test_fraction"]

    if not sessions:
        raise ValueError("No sessions provided for training data creation")

    # Ensure test_fraction has a valid value
    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError("test_fraction must be a float between 0 and 1")

    all_data_list = []
    for session in sessions:
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        # Format the data for the RNN model
        session_array = format_xarray_for_rnn(
            ds=ds,
            read_from_variable=read_from_variable,
        )
        all_data_list.append(session_array)

    if split_mode == "mode_1":
        # Original mode: Take initial portion of combined data
        all_data_array = np.concatenate(all_data_list, axis=1)
        test_size = int(all_data_array.shape[1] * test_fraction)
        data_test = all_data_array[:, :test_size]
        data_train = all_data_array[:, test_size:]
        logger.info(f"Mode 1 split - Initial {test_fraction:.1%} of combined data used for testing")

    else:  # mode_2
        # New mode: Take random continuous chunks from each session
        test_chunks: List[np.ndarray] = []
        train_chunks: List[np.ndarray] = []

        for session_idx, session_array in enumerate(all_data_list):
            session_name = sessions[session_idx]
            # Calculate test chunk size for this session
            session_length = session_array.shape[1]
            test_size = int(session_length * test_fraction)

            # Randomly select start index for test chunk
            max_start = session_length - test_size
            test_start = np.random.randint(0, max_start)
            test_end = test_start + test_size

            # Split into test and train chunks
            test_chunk = session_array[:, test_start:test_end]
            train_chunk_1 = session_array[:, :test_start]
            train_chunk_2 = session_array[:, test_end:]

            # Add to respective lists
            test_chunks.append(test_chunk)
            if train_chunk_1.shape[1] > 0:  # Only append non-empty chunks
                train_chunks.append(train_chunk_1)
            if train_chunk_2.shape[1] > 0:
                train_chunks.append(train_chunk_2)

            logger.info(f"Session {session_name}: test chunk {test_start}:{test_end} (length {test_size})")

        # Concatenate all chunks
        data_test = np.concatenate(test_chunks, axis=1)
        data_train = np.concatenate(train_chunks, axis=1)

    # Create train directory if it doesn't exist
    train_dir = Path(project_path) / "data" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays with the test/train info:
    train_data_path = train_dir / "train_seq.npy"
    np.save(str(train_data_path), data_train)

    test_data_path = train_dir / "test_seq.npy"
    np.save(str(test_data_path), data_test)

    logger.info(f"Length of train data: {data_train.shape[1]}")
    logger.info(f"Length of test data: {data_test.shape[1]}")


# def traindata_fixed(
#     config: dict,
#     sessions: List[str],
#     testfraction: float,
#     num_features: int,
#     savgol_filter: bool,
#     check_parameter: bool,
#     pose_ref_index: Optional[List[int]],
# ) -> None:
#     """
#     Create training dataset for fixed data.

#     Parameters
#     ---------
#     config : dict
#         Configuration parameters.
#     sessions : List[str]
#         List of sessions.
#     testfraction : float
#         Fraction of data to use as test data.
#     num_features : int
#         Number of features.
#     savgol_filter : bool
#         Flag indicating whether to apply Savitzky-Golay filter.
#     check_parameter : bool
#         If True, the function will plot the z-scored data and the filtered data.
#     pose_ref_index : Optional[List[int]]
#         List of reference coordinate indices for alignment.

#     Returns
#         None
#             Save numpy arrays with the test/train info to the project folder.
#     """
#     X_train = []
#     pos = []
#     pos_temp = 0
#     pos.append(0)

#     if check_parameter:
#         X_true = []
#         sessions = [sessions[0]]

#     for session in sessions:
#         logger.info("z-scoring of file %s" % session)
#         path_to_file = os.path.join(
#             config["project_path"],
#             "data",
#             "processed",
#             session,
#             session + "-PE-seq.npy",
#         )
#         data = np.load(path_to_file)

#         X_mean = np.mean(data, axis=None)
#         X_std = np.std(data, axis=None)
#         X_z = (data.T - X_mean) / X_std

#         if check_parameter:
#             X_z_copy = X_z.copy()
#             X_true.append(X_z_copy)

#         if config["robust"]:
#             iqr_val = iqr(X_z)
#             logger.info("IQR value: %.2f, IQR cutoff: %.2f" % (iqr_val, config["iqr_factor"] * iqr_val))
#             for i in range(X_z.shape[0]):
#                 for marker in range(X_z.shape[1]):
#                     if X_z[i, marker] > config["iqr_factor"] * iqr_val:
#                         X_z[i, marker] = np.nan

#                     elif X_z[i, marker] < -config["iqr_factor"] * iqr_val:
#                         X_z[i, marker] = np.nan

#                 X_z[i, :] = interpol_all_nans(X_z[i, :])

#         X_len = len(data.T)
#         pos_temp += X_len
#         pos.append(pos_temp)
#         X_train.append(X_z)

#     X = np.concatenate(X_train, axis=0).T

#     if savgol_filter:
#         X_med = scipy.signal.savgol_filter(X, config["savgol_length"], config["savgol_order"])
#     else:
#         X_med = X

#     num_frames = len(X_med.T)
#     test = int(num_frames * testfraction)

#     z_test = X_med[:, :test]
#     z_train = X_med[:, test:]

#     if check_parameter:
#         plot_check_parameter(
#             config,
#             iqr_val,
#             num_frames,
#             X_true,
#             X_med,
#         )

#     else:
#         if pose_ref_index is None:
#             raise ValueError("Please provide a pose reference index for training on fixed data. E.g. [0,5]")
#         # save numpy arrays the the test/train info:
#         np.save(
#             os.path.join(
#                 config["project_path"],
#                 "data",
#                 "train",
#                 "train_seq.npy",
#             ),
#             z_train,
#         )
#         np.save(
#             os.path.join(
#                 config["project_path"],
#                 "data",
#                 "train",
#                 "test_seq.npy",
#             ),
#             z_test,
#         )

#         y_shifted_indices = np.arange(0, num_features, 2)
#         x_shifted_indices = np.arange(1, num_features, 2)
#         belly_Y_ind = pose_ref_index[0] * 2
#         belly_X_ind = (pose_ref_index[0] * 2) + 1

#         for i, session in enumerate(sessions):
#             # Shifting section added 2/29/2024 PN
#             X_med_shifted_file = X_med[:, pos[i] : pos[i + 1]]
#             belly_Y_shift = X_med[belly_Y_ind, pos[i] : pos[i + 1]]
#             belly_X_shift = X_med[belly_X_ind, pos[i] : pos[i + 1]]

#             X_med_shifted_file[y_shifted_indices, :] -= belly_Y_shift
#             X_med_shifted_file[x_shifted_indices, :] -= belly_X_shift

#             np.save(
#                 os.path.join(
#                     config["project_path"],
#                     "data",
#                     "processed",
#                     session,
#                     session + "-PE-seq-clean.npy",
#                 ),
#                 X_med_shifted_file,
#             )  # saving new shifted file

#         logger.info("Lenght of train data: %d" % len(z_train.T))
#         logger.info("Lenght of test data: %d" % len(z_test.T))


@save_state(model=CreateTrainsetFunctionSchema)
def create_trainset(
    config: dict,
    test_fraction: float = 0.1,
    read_from_variable: str = "position_processed",
    split_mode: Literal["mode_1", "mode_2"] = "mode_2",
    save_logs: bool = True,
) -> None:
    """
    Creates training and test datasets for the VAME model.
    Fills in the values in the "create_trainset" key of the states.json file.
    Creates the training dataset for VAME at:
    - project_name/
        - data/
            - session00/
                - session00-PE-seq-clean.npy
            - session01/
                - session01-PE-seq-clean.npy
            - train/
                - test_seq.npy
                - train_seq.npy

    The produced -clean.npy files contain the aligned time series data in the
    shape of (num_dlc_features - 2, num_video_frames).

    The produced test_seq.npy contains the combined data in the shape of (num_dlc_features - 2, num_video_frames * test_fraction).

    The produced train_seq.npy contains the combined data in the shape of (num_dlc_features - 2, num_video_frames * (1 - test_fraction)).

    Parameters
    ----------
    config : dict
        Configuration parameters dictionary.
    test_fraction : float, optional
        Fraction of data to use as test data. Defaults to 0.1.
    read_from_variable : str, optional
        Variable name to read from the processed data. Defaults to "position_processed".
    split_mode : Literal["mode_1", "mode_2"], optional
        Mode for splitting data into train/test sets:
        - mode_1: Original mode that takes the initial test_fraction portion of the combined data
                 for testing and the rest for training.
        - mode_2: Takes random continuous chunks from each session proportional to test_fraction
                 for testing and uses the remaining parts for training.
        Defaults to "mode_2".
    save_logs : bool, optional
        Whether to save logs. Defaults to True.

    Returns
    -------
    None
    """
    try:
        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "create_trainset.log"
            logger_config.add_file_handler(str(log_path))

        if not os.path.exists(os.path.join(config["project_path"], "data", "train", "")):
            os.mkdir(os.path.join(config["project_path"], "data", "train", ""))

        fixed = config["egocentric_data"]

        sessions = []
        if config["all_data"] == "No":
            for session in config["session_names"]:
                use_session = input("Do you want to train on " + session + "? yes/no: ")
                if use_session == "yes":
                    sessions.append(session)
                if use_session == "no":
                    continue
        else:
            sessions = config["session_names"]

        logger.info("Creating training dataset...")

        if not fixed:
            logger.info("Creating trainset from the vame.egocentrical_alignment() output ")
            traindata_aligned(
                config=config,
                sessions=sessions,
                test_fraction=test_fraction,
                read_from_variable=read_from_variable,
                split_mode=split_mode,
            )
        else:
            raise NotImplementedError("Fixed data training is not implemented yet")
            # logger.info("Creating trainset from the vame.pose_to_numpy() output ")
            # traindata_fixed(
            #     config,
            #     sessions,
            #     config["test_fraction"],
            #     config["num_features"],
            #     config["savgol_filter"],
            #     check_parameter,
            #     pose_ref_index,
            # )

        logger.info("A training and test set has been created. Next step: vame.train_model()")

    except Exception as e:
        logger.exception(str(e))
        raise e
    finally:
        logger_config.remove_file_handler()
