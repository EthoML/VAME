import json
import numpy as np
from pathlib import Path
from typing import List, Literal
import datetime

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
    session_metadata = None

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

        # Format the data for the RNN model and get metadata
        session_array, metadata = format_xarray_for_rnn(
            ds=ds,
            read_from_variable=read_from_variable,
            keypoints=keypoints,
        )
        all_data_list.append(session_array)

        # Store metadata from first session (should be consistent across sessions)
        if session_metadata is None:
            session_metadata = metadata

    # Ensure we have metadata
    if session_metadata is None:
        raise ValueError("No metadata collected from sessions")

    # Track split details for metadata
    split_details = {}

    if split_mode == "mode_1":
        # Original mode: Take initial portion of combined data
        all_data_array = np.concatenate(all_data_list, axis=1)
        test_size = int(all_data_array.shape[1] * test_fraction)
        data_test = all_data_array[:, :test_size]
        data_train = all_data_array[:, test_size:]

        # Track split details
        split_details = {
            "method": "sequential_split",
            "split_index": test_size,
            "total_length": all_data_array.shape[1],
            "session_boundaries": []
        }

        # Calculate session boundaries in concatenated array
        cumulative_length = 0
        for i, session_array in enumerate(all_data_list):
            session_length = session_array.shape[1]
            split_details["session_boundaries"].append({
                "session": sessions[i],
                "start": cumulative_length,
                "end": cumulative_length + session_length,
                "length": session_length
            })
            cumulative_length += session_length

        logger.info(f"Mode 1 split - Initial {test_fraction:.1%} of combined data used for testing")

    else:  # mode_2
        # New mode: Take random continuous chunks from each session
        test_chunks: List[np.ndarray] = []
        train_chunks: List[np.ndarray] = []
        session_splits = []

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

            # Track split details for this session
            train_chunks_info = []
            if train_chunk_1.shape[1] > 0:
                train_chunks_info.append({"start": 0, "end": test_start})
            if train_chunk_2.shape[1] > 0:
                train_chunks_info.append({"start": test_end, "end": session_length})

            session_splits.append({
                "session": session_name,
                "session_length": session_length,
                "test_chunk": {"start": test_start, "end": test_end},
                "train_chunks": train_chunks_info
            })

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

        split_details = {
            "method": "random_continuous_chunks",
            "session_splits": session_splits
        }

    # Create train directory if it doesn't exist
    train_dir = Path(project_path) / "data" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays with the test/train info:
    train_data_path = train_dir / "train_seq.npy"
    np.save(str(train_data_path), data_train)

    test_data_path = train_dir / "test_seq.npy"
    np.save(str(test_data_path), data_test)

    # Create and save single metadata file for provenance tracking
    metadata = {
        "feature_mapping": session_metadata["feature_mapping"],
        "parameters": session_metadata["parameters"],
        "split_information": {
            "split_mode": split_mode,
            "test_fraction": test_fraction,
            "sessions_used": sessions,
            "split_details": split_details
        },
        "data_info": {
            "train": {
                "shape": data_train.shape,
                "total_samples": data_train.shape[0],
                "features": data_train.shape[1]
            },
            "test": {
                "shape": data_test.shape,
                "total_samples": data_test.shape[0],
                "features": data_test.shape[1]
            }
        },
        "creation_timestamp": datetime.datetime.now().isoformat(),
        "vame_version": config["vame_version"],
    }

    # Save single metadata file
    metadata_path = train_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata file saved for feature provenance tracking")
    logger.info(f"Metadata: {metadata_path}")

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
                - metadata.json

    The produced test_seq.npy contains the combined data in the shape of (num_features - 3, num_video_frames * test_fraction).
    The produced train_seq.npy contains the combined data in the shape of (num_features - 3, num_video_frames * (1 - test_fraction)).
    The metadata.json file contains feature provenance information for tracking which keypoints and coordinates
    correspond to each feature in the numpy arrays, along with detailed split information for full reproducibility.

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
