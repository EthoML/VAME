import os
import numpy as np
import pandas as pd
from pathlib import Path

from vame.logging.logger import VameLogger
from vame.util.data_manipulation import interpol_first_rows_nans
from vame.io.load_poses import read_pose_estimation_file


logger_config = VameLogger(__name__)
logger = logger_config.logger


def pose_to_numpy(
    config: dict,
    save_logs=False,
) -> None:
    """
    Converts a pose-estimation.csv file to a numpy array.
    Note that this code is only useful for data which is a priori egocentric, i.e. head-fixed
    or otherwise restrained animals.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    save_logs : bool, optional
        If True, the logs will be saved to a file, by default False.

    Raises
    ------
    ValueError
        If the config.yaml file indicates that the data is not egocentric.
    """
    try:
        if save_logs:
            log_path = Path(config["project_path"]) / "logs" / "pose_to_numpy.log"
            logger_config.add_file_handler(str(log_path))

        project_path = config["project_path"]
        sessions = config["session_names"]
        confidence = config["pose_confidence"]
        if not config["egocentric_data"]:
            raise ValueError(
                "The config.yaml indicates that the data is not egocentric. Please check the parameter egocentric_data"
            )

        file_type = config["pose_estimation_filetype"]
        paths_to_pose_nwb_series_data = config["paths_to_pose_nwb_series_data"]
        for i, session in enumerate(sessions):
            file_path = os.path.join(
                project_path,
                "data",
                "raw",
                session + ".nc",
            )
            data, data_mat, ds = read_pose_estimation_file(
                file_path=file_path,
                file_type=file_type,
                path_to_pose_nwb_series_data=(
                    paths_to_pose_nwb_series_data
                    if not paths_to_pose_nwb_series_data
                    else paths_to_pose_nwb_series_data[i]
                ),
            )

            pose_list = []

            # get the number of bodyparts, their x,y-position and the confidence from DeepLabCut
            for i in range(int(data_mat.shape[1] / 3)):
                pose_list.append(data_mat[:, i * 3 : (i + 1) * 3])

            # find low confidence and set them to NaN
            for i in pose_list:
                for j in i:
                    if j[2] <= confidence:
                        j[0], j[1] = np.nan, np.nan

            # interpolate NaNs
            for i in pose_list:
                i = interpol_first_rows_nans(i)

            positions = np.concatenate(pose_list, axis=1)
            final_positions = np.zeros((data_mat.shape[0], int(data_mat.shape[1] / 3) * 2))

            jdx = 0
            idx = 0
            for i in range(int(data_mat.shape[1] / 3)):
                final_positions[:, idx : idx + 2] = positions[:, jdx : jdx + 2]
                jdx += 3
                idx += 2

            # save the final_positions array with np.save()
            np.save(
                os.path.join(
                    project_path,
                    "data",
                    "processed",
                    session + "-PE-seq.npy",
                ),
                final_positions.T,
            )
            logger.info("conversion from DeepLabCut csv to numpy complete...")

        logger.info("Your data is now in right format and you can call vame.create_trainset()")
    except Exception as e:
        logger.exception(f"{e}")
        raise e
    finally:
        logger_config.remove_file_handler()
