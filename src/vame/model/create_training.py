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
    keypoints_to_include: List[str] | None = None,
    keypoints_to_exclude: List[str] | None = None,
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

    if keypoints_to_include and keypoints_to_exclude:
        raise ValueError("Cannot specify both keypoints_to_include and keypoints_to_exclude. Choose one.")

    # Ensure test_fraction has a valid value
    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError("test_fraction must be a float between 0 and 1")

    all_data_list = []
    for session in sessions:
        # Read session data
        file_path = str(Path(project_path) / "data" / "processed" / f"{session}_processed.nc")
        _, _, ds = read_pose_estimation_file(file_path=file_path)

        keypoints = ds.keypoints.values
        if keypoints_to_include is not None:
            if any(k not in keypoints for k in keypoints_to_include):
                raise ValueError("Some keypoints in `keypoints_to_include` are not present in the dataset.")
            keypoints = keypoints_to_include
        elif keypoints_to_exclude is not None:
            if any(k not in keypoints for k in keypoints_to_exclude):
                raise ValueError("Some keypoints in `keypoints_to_exclude` are not present in the dataset.")
            keypoints = [k for k in keypoints if k not in keypoints_to_exclude]

        # Format the data for the RNN model
        session_array = format_xarray_for_rnn(
            ds=ds,
            read_from_variable=read_from_variable,
            keypoints=keypoints,
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


@save_state(model=CreateTrainsetFunctionSchema)
def create_trainset(
    config: dict,
    test_fraction: float = 0.1,
    read_from_variable: str = "position_processed",
    split_mode: Literal["mode_1", "mode_2"] = "mode_2",
    keypoints_to_include: List[str] | None = None,
    keypoints_to_exclude: List[str] | None = None,
    save_logs: bool = True,
) -> None:
    """
    Creates training and test datasets for the VAME model.
    Fills in the values in the "create_trainset" key of the states.json file.
    Creates the training dataset for VAME at:
    - project_name/
        - data/
            - train/
                - test_seq.npy
                - train_seq.npy

    The produced test_seq.npy contains the combined data in the shape of (num_features - 3, num_video_frames * test_fraction).
    The produced train_seq.npy contains the combined data in the shape of (num_features - 3, num_video_frames * (1 - test_fraction)).

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
            traindata_aligned(
                config=config,
                sessions=sessions,
                test_fraction=test_fraction,
                read_from_variable=read_from_variable,
                split_mode=split_mode,
                keypoints_to_include=keypoints_to_include,
                keypoints_to_exclude=keypoints_to_exclude,
            )
        else:
            raise NotImplementedError("Fixed data training is not implemented yet")

        logger.info("A training and test set has been created. Next step: vame.train_model()")

    except Exception as e:
        logger.exception(str(e))
        raise e
    finally:
        logger_config.remove_file_handler()
