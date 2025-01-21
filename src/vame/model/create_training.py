import os
import numpy as np
from pathlib import Path
from typing import List

from vame.logging.logger import VameLogger
from vame.schemas.states import CreateTrainsetFunctionSchema, save_state
from vame.io.load_poses import read_pose_estimation_file
from vame.preprocessing.to_model import format_xarray_for_rnn


logger_config = VameLogger(__name__)
logger = logger_config.logger


def traindata_aligned(
    config: dict,
    sessions: List[str] | None = None,
    test_fraction: float | None = None,
    read_from_variable: str = "position_processed",
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

    Returns
    -------
    None
    """
    project_path = config["project_path"]
    if sessions is None:
        sessions = config["session_names"]
    if test_fraction is None:
        test_fraction = config["test_fraction"]

    all_data_list = []
    for session in sessions:
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        # Format the data for the RNN model
        filtered_array = format_xarray_for_rnn(
            ds=ds,
            read_from_variable=read_from_variable,
        )

        all_data_list.append(filtered_array)

    all_data_array = np.concatenate(all_data_list, axis=1)
    test_size = int(all_data_array.shape[1] * test_fraction)
    data_test = all_data_array[:, :test_size]
    data_train = all_data_array[:, test_size:]

    # Save numpy arrays the the test/train info:
    train_data_path = Path(project_path) / "data" / "train" / "train_seq.npy"
    np.save(str(train_data_path), data_train)

    test_data_path = Path(project_path) / "data" / "train" / "test_seq.npy"
    np.save(str(test_data_path), data_test)

    logger.info(f"Lenght of train data: {data_train.shape[1]}")
    logger.info(f"Lenght of test data: {data_test.shape[1]}")


# def traindata_fixed(
#     cfg: dict,
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
#     cfg : dict
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
#             cfg["project_path"],
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

#         if cfg["robust"]:
#             iqr_val = iqr(X_z)
#             logger.info("IQR value: %.2f, IQR cutoff: %.2f" % (iqr_val, cfg["iqr_factor"] * iqr_val))
#             for i in range(X_z.shape[0]):
#                 for marker in range(X_z.shape[1]):
#                     if X_z[i, marker] > cfg["iqr_factor"] * iqr_val:
#                         X_z[i, marker] = np.nan

#                     elif X_z[i, marker] < -cfg["iqr_factor"] * iqr_val:
#                         X_z[i, marker] = np.nan

#                 X_z[i, :] = interpol_all_nans(X_z[i, :])

#         X_len = len(data.T)
#         pos_temp += X_len
#         pos.append(pos_temp)
#         X_train.append(X_z)

#     X = np.concatenate(X_train, axis=0).T

#     if savgol_filter:
#         X_med = scipy.signal.savgol_filter(X, cfg["savgol_length"], cfg["savgol_order"])
#     else:
#         X_med = X

#     num_frames = len(X_med.T)
#     test = int(num_frames * testfraction)

#     z_test = X_med[:, :test]
#     z_train = X_med[:, test:]

#     if check_parameter:
#         plot_check_parameter(
#             cfg,
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
#                 cfg["project_path"],
#                 "data",
#                 "train",
#                 "train_seq.npy",
#             ),
#             z_train,
#         )
#         np.save(
#             os.path.join(
#                 cfg["project_path"],
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
#                     cfg["project_path"],
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
    save_logs: bool = False,
) -> None:
    """
    Creates a training and test datasets for the VAME model.
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
    save_logs : bool, optional
        If True, the function will save logs to the project folder. Defaults to False.

    Returns
    -------
    None
    """
    try:
        fixed = config["egocentric_data"]

        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "create_trainset.log"
            logger_config.add_file_handler(str(log_path))

        if not os.path.exists(os.path.join(config["project_path"], "data", "train", "")):
            os.mkdir(os.path.join(config["project_path"], "data", "train", ""))

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
            )
        else:
            raise NotImplementedError("Fixed data training is not implemented yet")
            # logger.info("Creating trainset from the vame.pose_to_numpy() output ")
            # traindata_fixed(
            #     cfg,
            #     sessions,
            #     cfg["test_fraction"],
            #     cfg["num_features"],
            #     cfg["savgol_filter"],
            #     check_parameter,
            #     pose_ref_index,
            # )

        logger.info("A training and test set has been created. Next step: vame.train_model()")

    except Exception as e:
        logger.exception(str(e))
        raise e
    finally:
        logger_config.remove_file_handler()
