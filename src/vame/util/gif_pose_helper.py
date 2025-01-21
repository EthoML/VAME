import os
import tqdm
import cv2 as cv
import numpy as np
import pandas as pd

from vame.logging.logger import VameLogger
from vame.io.load_poses import read_pose_estimation_file
from vame.util.data_manipulation import (
    interpol_first_rows_nans,
    crop_and_flip_legacy,
    background,
)


logger_config = VameLogger(__name__)
logger = logger_config.logger


def get_animal_frames(
    cfg: dict,
    session: str,
    pose_ref_index: list,
    start: int,
    length: int,
    subtract_background: bool,
    file_format: str = ".mp4",
    crop_size: tuple = (300, 300),
) -> list:
    """
    Extracts frames of an animal from a video file and returns them as a list.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing project information.
    session : str
        Name of the session.
    pose_ref_index : list
        List of reference coordinate indices for alignment.
    start : int
        Starting frame index.
    length : int
        Number of frames to extract.
    subtract_background : bool
        Whether to subtract background or not.
    file_format : str, optional
        Format of the video file. Defaults to '.mp4'.
    crop_size : tuple, optional
        Size of the cropped area. Defaults to (300, 300).

    Returns
    -------
    list:
        List of extracted frames.
    """
    project_path = cfg["project_path"]
    time_window = cfg["time_window"]
    lag = int(time_window / 2)

    video_path = os.path.join(
        project_path,
        "data",
        "raw",
        session + file_format,
    )

    # read out data
    # data = pd.read_csv(
    #     os.path.join(
    #         path_to_file,
    #         "videos",
    #         "pose_estimation",
    #         session + ".csv",
    #     ),
    #     skiprows=2,
    # )
    # data_mat = pd.DataFrame.to_numpy(data)
    # data_mat = data_mat[:, 1:]

    file_path = os.path.join(
        project_path,
        "data",
        "raw",
        session + ".nc",
    )
    data, data_mat, ds = read_pose_estimation_file(file_path=file_path)

    # get the coordinates for alignment from data table
    pose_list = []

    for i in range(int(data_mat.shape[1] / 3)):
        pose_list.append(data_mat[:, i * 3 : (i + 1) * 3])

    # list of reference coordinate indices for alignment
    # 0: snout, 1: forehand_left, 2: forehand_right,
    # 3: hindleft, 4: hindright, 5: tail

    pose_ref_index = pose_ref_index

    # list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = pose_ref_index

    # compute background
    if subtract_background:
        try:
            logger.info("Loading background image ...")
            bg = np.load(
                os.path.join(
                    project_path,
                    "data",
                    "processed",
                    session + "-background.npy",
                )
            )
        except Exception:
            logger.info("Can't find background image... Calculate background image...")
            bg = background(
                project_path=project_path,
                session=session,
                video_path=video_path,
                save_background=True,
            )

    images = []
    points = []

    for i in pose_list:
        for j in i:
            if j[2] <= 0.8:
                j[0], j[1] = np.nan, np.nan

    for i in pose_list:
        i = interpol_first_rows_nans(i)

    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        raise Exception(f"Unable to open video file: {video_path}")

    for idx in tqdm.tqdm(range(length), disable=not True, desc="Align frames"):
        try:
            capture.set(1, idx + start + lag)
            ret, frame = capture.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if subtract_background:
                frame = frame - bg
                frame[frame <= 0] = 0
        except Exception:
            logger.info(f"Couldn't find a frame in capture.read(). #Frame: {idx + start + lag}")
            continue

        # Read coordinates and add border
        pose_list_bordered = []

        for i in pose_list:
            pose_list_bordered.append(
                (
                    int(i[idx + start + lag][0] + crop_size[0]),
                    int(i[idx + start + lag][1] + crop_size[1]),
                )
            )

        img = cv.copyMakeBorder(
            frame,
            crop_size[1],
            crop_size[1],
            crop_size[0],
            crop_size[0],
            cv.BORDER_CONSTANT,
            0,
        )

        punkte = []
        for i in pose_ref_index:
            coord = []
            coord.append(pose_list_bordered[i][0])
            coord.append(pose_list_bordered[i][1])
            punkte.append(coord)
        punkte = [punkte]
        punkte = np.asarray(punkte)

        # calculate minimal rectangle around snout and tail
        rect = cv.minAreaRect(punkte)

        # change size in rect tuple structure to be equal to crop_size
        lst = list(rect)
        lst[1] = crop_size
        rect = tuple(lst)

        center, size, theta = rect

        # crop image
        out, shifted_points = crop_and_flip_legacy(
            rect,
            img,
            pose_list_bordered,
            pose_flip_ref,
        )
        images.append(out)
        points.append(shifted_points)

    capture.release()
    return images
