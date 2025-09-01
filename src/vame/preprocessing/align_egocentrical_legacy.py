# import os
# import cv2 as cv
# import numpy as np
# import pandas as pd
# import tqdm
# from typing import Tuple, List, Union
# from pathlib import Path

# from vame.logging.logger import VameLogger, TqdmToLogger
# from vame.util.auxiliary import read_config
# from vame.schemas.states import EgocentricAlignmentFunctionSchema, save_state
# from vame.schemas.project import PoseEstimationFiletype
# from vame.io.load_poses import read_pose_estimation_file
# from vame.util.data_manipulation import (
#     interpol_first_rows_nans,
#     crop_and_flip_legacy,
#     background,
# )
# from vame.video import get_video_frame_rate


# logger_config = VameLogger(__name__)
# logger = logger_config.logger


# def align_mouse_legacy(
#     project_path: str,
#     session: str,
#     video_format: str,
#     crop_size: Tuple[int, int],
#     pose_list: List[np.ndarray],
#     pose_ref_index: Tuple[int, int],
#     confidence: float,
#     pose_flip_ref: Tuple[int, int],
#     bg: np.ndarray,
#     frame_count: int,
#     use_video: bool = True,
#     tqdm_stream: Union[TqdmToLogger, None] = None,
# ) -> Tuple[List[np.ndarray], List[List[np.ndarray]], np.ndarray]:
#     """
#     Align the mouse in the video frames.

#     Parameters
#     ----------
#     project_path : str
#         Path to the project directory.
#     session : str
#         Name of the session.
#     video_format : str
#         Format of the video file.
#     crop_size : Tuple[int, int]
#         Size to crop the video frames.
#     pose_list : List[np.ndarray]
#         List of pose coordinates.
#     pose_ref_index : Tuple[int, int]
#         Pose reference indices.
#     confidence : float
#         Pose confidence threshold.
#     pose_flip_ref : Tuple[int, int]
#         Reference indices for flipping.
#     bg : np.ndarray
#         Background image.
#     frame_count : int
#         Number of frames to align.
#     use_video : bool, optional
#         bool if video should be cropped or DLC points only. Defaults to True.
#     tqdm_stream : Union[TqdmToLogger, None], optional
#         Tqdm stream to log the progress. Defaults to None.

#     Returns
#     -------
#     Tuple[List[np.ndarray], List[List[np.ndarray]], np.ndarray]
#         List of aligned images, list of aligned DLC points, and aligned time series data.
#     """
#     images = []
#     points = []
#     for i in pose_list:
#         for j in i:
#             if j[2] <= confidence:
#                 j[0], j[1] = np.nan, np.nan

#     for i in pose_list:
#         i = interpol_first_rows_nans(i)

#     if use_video:
#         video_path = str(
#             os.path.join(
#                 project_path,
#                 "data",
#                 "raw",
#                 session + video_format,
#             )
#         )
#         capture = cv.VideoCapture(video_path)
#         if not capture.isOpened():
#             raise Exception(f"Unable to open video file: {video_path}")

#     for idx in tqdm.tqdm(
#         range(frame_count),
#         disable=not True,
#         file=tqdm_stream,
#         desc="Align frames",
#     ):
#         if use_video:
#             # Read frame
#             try:
#                 ret, frame = capture.read()
#                 frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#                 frame = frame - bg
#                 frame[frame <= 0] = 0
#             except Exception:
#                 logger.info("Couldn't find a frame in capture.read(). #Frame: %d" % idx)
#                 continue
#         else:
#             frame = np.zeros((1, 1))

#         # Read coordinates and add border
#         pose_list_bordered = []

#         for i in pose_list:
#             pose_list_bordered.append((int(i[idx][0] + crop_size[0]), int(i[idx][1] + crop_size[1])))

#         img = cv.copyMakeBorder(
#             frame,
#             crop_size[1],
#             crop_size[1],
#             crop_size[0],
#             crop_size[0],
#             cv.BORDER_CONSTANT,
#             0,
#         )

#         coord_center = []
#         punkte = []

#         for i in pose_ref_index:
#             coord = []
#             # changed from pose_list_bordered[i][0] 2/28/2024 PN
#             coord.append(pose_list_bordered[i][0])
#             # changed from pose_list_bordered[i][1] 2/28/2024 PN
#             coord.append(pose_list_bordered[i][1])
#             punkte.append(coord)

#         # coord_center.append(pose_list_bordered[5][0]-5)
#         # coord_center.append(pose_list_bordered[5][0]+5)
#         # coord_center = [coord_center]
#         punkte = [punkte]

#         # coord_center = np.asarray(coord_center)
#         punkte = np.asarray(punkte)

#         # calculate minimal rectangle around snout and tail
#         rect = cv.minAreaRect(punkte)
#         # rect_belly = cv.minAreaRect(coord_center)
#         # center_belly, size_belly, theta_belly = rect_belly
#         # change size in rect tuple structure to be equal to crop_size
#         lst = list(rect)
#         lst[1] = crop_size
#         # lst[0] = center_belly
#         rect = tuple(lst)
#         center, size, theta = rect

#         # crop image
#         out, shifted_points = crop_and_flip_legacy(
#             rect,
#             img,
#             pose_list_bordered,
#             pose_flip_ref,
#         )

#         if use_video:  # for memory optimization, just save images when video is used.
#             images.append(out)
#         points.append(shifted_points)

#     if use_video:
#         capture.release()

#     time_series = np.zeros((len(pose_list) * 2, frame_count))
#     for i in range(frame_count):
#         idx = 0
#         for j in range(len(pose_list)):
#             time_series[idx : idx + 2, i] = points[i][j]
#             idx += 2

#     return images, points, time_series


# def alignment_legacy(
#     project_path: str,
#     session: str,
#     pose_ref_index: Tuple[int, int],
#     video_format: str,
#     crop_size: Tuple[int, int],
#     confidence: float,
#     pose_estimation_filetype: PoseEstimationFiletype,
#     path_to_pose_nwb_series_data: Union[str, None] = None,
#     use_video: bool = False,
#     tqdm_stream: Union[TqdmToLogger, None] = None,
# ) -> Tuple[np.ndarray, List[np.ndarray]]:
#     """
#     Perform alignment of egocentric data.

#     Parameters
#     ----------
#     project_path : str
#         Path to the project directory.
#     session : str
#         Name of the session.
#     pose_ref_index : List[int]
#         Pose reference indices.
#     video_format : str
#         Format of the video file.
#     crop_size : Tuple[int, int]
#         Size to crop the video frames.
#     confidence : float
#         Pose confidence threshold.
#     pose_estimation_filetype : PoseEstimationFiletype
#         Pose estimation file type. Can be .csv or .nwb.
#     path_to_pose_nwb_series_data : Union[str, None], optional
#         Path to the pose series data in nwb files. Defaults to None.
#     use_video : bool, optional
#         Whether to use video for alignment. Defaults to False.
#     tqdm_stream : Union[TqdmToLogger, None], optional
#         Tqdm stream to log the progress. Defaults to None.

#     Returns
#     -------
#     Tuple[np.ndarray, List[np.ndarray]]
#         Aligned time series data and list of aligned frames.
#     """
#     # read out data
#     file_path = str(Path(project_path) / "data" / "raw" / f"{session}.nc")
#     data, data_mat, _ = read_pose_estimation_file(
#         file_path=file_path,
#         file_type=pose_estimation_filetype,
#         path_to_pose_nwb_series_data=path_to_pose_nwb_series_data,
#     )

#     # get the coordinates for alignment from data table
#     # pose_list dimensions: (num_body_parts, num_frames, 3)
#     pose_list = []
#     for i in range(int(data_mat.shape[1] / 3)):
#         pose_list.append(data_mat[:, i * 3 : (i + 1) * 3])

#     # list of reference coordinate indices for alignment
#     # 0: snout, 1: forehand_left, 2: forehand_right,
#     # 3: hindleft, 4: hindright, 5: tail
#     # list of 2 reference coordinate indices for avoiding flipping
#     pose_flip_ref = pose_ref_index

#     if use_video:
#         # compute background
#         video_path = str(
#             os.path.join(
#                 project_path,
#                 "data",
#                 "raw",
#                 session + video_format,
#             )
#         )
#         bg = background(
#             project_path=project_path,
#             session=session,
#             video_path=video_path,
#             save_background=False,
#         )
#         frame_count = get_video_frame_rate(video_path)
#     else:
#         bg = 0
#         # Change this to an abitrary number if you first want to test the code
#         frame_count = len(data)

#     frames, n, time_series = align_mouse_legacy(
#         project_path=project_path,
#         session=session,
#         video_format=video_format,
#         crop_size=crop_size,
#         pose_list=pose_list,
#         pose_ref_index=pose_ref_index,
#         confidence=confidence,
#         pose_flip_ref=pose_flip_ref,
#         bg=bg,
#         frame_count=frame_count,
#         use_video=use_video,
#         tqdm_stream=tqdm_stream,
#     )

#     return time_series, frames


# # @save_state(model=EgocentricAlignmentFunctionSchema)
# def egocentric_alignment_legacy(
#     config: str,
#     pose_ref_index: Tuple[int, int] = (0, 1),
#     crop_size: Tuple[int, int] = (300, 300),
#     use_video: bool = False,
#     video_format: str = ".mp4",
#     check_video: bool = False,
#     save_logs: bool = False,
# ) -> None:
#     """
#     Egocentric alignment of bevarioral videos.
#     Fills in the values in the "egocentric_alignment" key of the states.json file.
#     Creates training dataset for VAME at:
#     - project_name/
#         - data/
#             - filename/
#                 - filename-PE-seq.npy
#             - filename/
#                 - filename-PE-seq.npy
#     The produced .npy files contain the aligned time series data in the
#     shape of (num_dlc_features, num_video_frames).

#     Parameters
#     ---------
#     config : str
#         Path for the project config file.
#     pose_ref_index : list, optional
#         Pose reference index to be used to align. Defaults to [0, 1].
#     crop_size : tuple, optional
#         Size to crop the video. Defaults to (300,300).
#     use_video : bool, optional
#         Weather to use video to do the post alignment. Defaults to False.
#     video_format : str, optional
#         Video format, can be .mp4 or .avi. Defaults to '.mp4'.
#     check_video : bool, optional
#         Weather to check the video. Defaults to False.

#     Raises:
#     ------
#     ValueError
#         If the config.yaml indicates that the data is not egocentric.
#     """
#     try:
#         config_file = Path(config).resolve()
#         config = read_config(str(config_file))
#         if config["egocentric_data"]:
#             raise ValueError(
#                 "The config.yaml indicates that the data is egocentric. Please check the parameter 'egocentric_data'."
#             )
#         tqdm_stream = None

#         if save_logs:
#             log_path = Path(config["project_path"]) / "logs" / "egocentric_alignment.log"
#             logger_config.add_file_handler(str(log_path))
#             tqdm_stream = TqdmToLogger(logger=logger)

#         logger.info("Starting egocentric alignment")
#         project_path = config["project_path"]
#         sessions = config["session_names"]
#         confidence = config["pose_confidence"]
#         num_features = config["num_features"]
#         video_format = video_format
#         crop_size = crop_size

#         y_shifted_indices = np.arange(0, num_features, 2)
#         x_shifted_indices = np.arange(1, num_features, 2)
#         belly_Y_ind = pose_ref_index[0] * 2
#         belly_X_ind = (pose_ref_index[0] * 2) + 1

#         # call function and save into your VAME data folder
#         paths_to_pose_nwb_series_data = config["paths_to_pose_nwb_series_data"]
#         for i, session in enumerate(sessions):
#             logger.info("Aligning session %s, Pose confidence value: %.2f" % (session, confidence))
#             egocentric_time_series, frames = alignment_legacy(
#                 project_path=project_path,
#                 session=session,
#                 pose_ref_index=pose_ref_index,
#                 video_format=video_format,
#                 crop_size=crop_size,
#                 confidence=confidence,
#                 pose_estimation_filetype=config["pose_estimation_filetype"],
#                 path_to_pose_nwb_series_data=(
#                     paths_to_pose_nwb_series_data
#                     if not paths_to_pose_nwb_series_data
#                     else paths_to_pose_nwb_series_data[i]
#                 ),
#                 use_video=use_video,
#                 tqdm_stream=tqdm_stream,
#             )

#             # Shifiting section added 2/29/2024 PN
#             # TODO - should this be hardcoded like that?
#             egocentric_time_series_shifted = egocentric_time_series
#             belly_Y_shift = egocentric_time_series[belly_Y_ind, :]
#             belly_X_shift = egocentric_time_series[belly_X_ind, :]

#             egocentric_time_series_shifted[y_shifted_indices, :] -= belly_Y_shift
#             egocentric_time_series_shifted[x_shifted_indices, :] -= belly_X_shift

#             # Save new shifted file
#             np.save(
#                 os.path.join(
#                     project_path,
#                     "data",
#                     "processed",
#                     session,
#                     session + "-PE-seq-legacy.npy",
#                 ),
#                 egocentric_time_series_shifted,
#             )

#         logger.info("Your data is now in the right format and you can call vame.create_trainset()")
#     except Exception as e:
#         logger.exception(f"{e}")
#         raise e
#     finally:
#         logger_config.remove_file_handler()


# @save_state(model=EgocentricAlignmentFunctionSchema)
# def egocentric_alignment(
#     config: str,
#     pose_ref_1: str = "snout",
#     pose_ref_2: str = "tailbase",
#     crop_size: Tuple[int, int] = (300, 300),
#     save_logs: bool = False,
# ) -> None:
#     """
#     Egocentric alignment of bevarioral videos.
#     Fills in the values in the "egocentric_alignment" key of the states.json file.
#     Creates training dataset for VAME at:
#     - project_name/
#         - data/
#             - filename/
#                 - filename-PE-seq.npy
#             - filename/
#                 - filename-PE-seq.npy
#     The produced .npy files contain the aligned time series data in the
#     shape of (num_dlc_features, num_video_frames).

#     Parameters
#     ---------
#     config : str
#         Path for the project config file.
#     pose_ref_index : list, optional
#         Pose reference index to be used to align. Defaults to [0, 1].
#     crop_size : tuple, optional
#         Size to crop the video. Defaults to (300,300).

#     Raises:
#     ------
#     ValueError
#         If the config.yaml indicates that the data is not egocentric.
#     """
#     try:
#         config_file = Path(config).resolve()
#         config = read_config(str(config_file))
#         if config["egocentric_data"]:
#             raise ValueError(
#                 "The config.yaml indicates that the data is egocentric. Please check the parameter 'egocentric_data'."
#             )
#         tqdm_stream = None

#         if save_logs:
#             log_path = Path(config["project_path"]) / "logs" / "egocentric_alignment.log"
#             logger_config.add_file_handler(str(log_path))
#             tqdm_stream = TqdmToLogger(logger=logger)

#         logger.info("Starting egocentric alignment")
#         project_path = config["project_path"]
#         sessions = config["session_names"]
#         confidence = config["pose_confidence"]
#         num_features = config["num_features"]

#         y_shifted_indices = np.arange(0, num_features, 2)
#         x_shifted_indices = np.arange(1, num_features, 2)
#         # reference_Y_ind = pose_ref_index[0] * 2
#         # reference_X_ind = (pose_ref_index[0] * 2) + 1

#         # call function and save into your VAME data folder
#         for i, session in enumerate(sessions):
#             logger.info("Aligning session %s, Pose confidence value: %.2f" % (session, confidence))
#             # read out data
#             file_path = str(Path(project_path) / "data" / "raw" / f"{session}.nc")
#             _, data_mat, ds = read_pose_estimation_file(file_path=file_path)

#             # get the coordinates for alignment from data table
#             # pose_list dimensions: (num_body_parts, num_frames, 3)
#             pose_list = []
#             for i in range(int(data_mat.shape[1] / 3)):
#                 pose_list.append(data_mat[:, i * 3 : (i + 1) * 3])

#             frame_count = ds.position.time.shape[0]
#             keypoints_names = ds.keypoints.values

#             reference_X_ind = np.where(ds.keypoints.values == pose_ref_1)[0][0] * 2
#             reference_Y_ind = reference_X_ind + 1

#             pose_ref_index = (
#                 np.where(keypoints_names == pose_ref_1)[0][0],
#                 np.where(keypoints_names == pose_ref_2)[0][0],
#             )

#             egocentric_time_series = alignment(
#                 crop_size=crop_size,
#                 pose_list=pose_list,
#                 pose_ref_index=pose_ref_index,
#                 confidence=confidence,
#                 frame_count=frame_count,
#                 tqdm_stream=tqdm_stream,
#             )

#             # Shifiting section added 2/29/2024 PN
#             egocentric_time_series_shifted = egocentric_time_series
#             reference_Y_shift = egocentric_time_series[reference_Y_ind, :]
#             reference_X_shift = egocentric_time_series[reference_X_ind, :]

#             egocentric_time_series_shifted[y_shifted_indices, :] -= reference_Y_shift
#             egocentric_time_series_shifted[x_shifted_indices, :] -= reference_X_shift

#             # Save new shifted file
#             np.save(
#                 os.path.join(
#                     project_path,
#                     "data",
#                     "processed",
#                     session,
#                     session + "-PE-seq.npy",
#                 ),
#                 egocentric_time_series_shifted,
#             )

#             # Add new variable to the dataset
#             ds["position_aligned"] = (
#                 ("time", "individuals", "keypoints", "space"),
#                 egocentric_time_series_shifted.T.reshape(frame_count, 1, len(keypoints_names), 2),
#             )
#             # save to file
#             result_file = Path(project_path) / "data" / "processed" / session / f"{session}-aligned.nc"
#             ds.to_netcdf(result_file, engine="netcdf4")

#         logger.info("Your data is now in the right format and you can call vame.create_trainset()")
#     except Exception as e:
#         logger.exception(f"{e}")
#         raise e
#     finally:
#         logger_config.remove_file_handler()


# def alignment(
#     crop_size: Tuple[int, int],
#     pose_list: List[np.ndarray],
#     pose_ref_index: Tuple[int, int],
#     confidence: float,
#     frame_count: int,
#     tqdm_stream: Union[TqdmToLogger, None] = None,
# ) -> np.ndarray:
#     """
#     Egocentric alignment of pose estimation data.

#     Parameters
#     ----------
#     crop_size : Tuple[int, int]
#         Size to crop the video frames.
#     pose_list : List[np.ndarray]
#         List of pose coordinates.
#     pose_ref_index : Tuple[int, int]
#         Pose reference indices.
#     confidence : float
#         Pose confidence threshold.
#     frame_count : int
#         Number of frames to align.
#     tqdm_stream : Union[TqdmToLogger, None], optional
#         Tqdm stream to log the progress. Defaults to None.

#     Returns
#     -------
#     np.ndarray
#         Aligned time series data.
#     """
#     points = []

#     # for i in pose_list:
#     #     for j in i:
#     #         if j[2] <= confidence:
#     #             j[0], j[1] = np.nan, np.nan

#     # for i in pose_list:
#     #     i = interpol_first_rows_nans(i)

#     for idx in tqdm.tqdm(
#         range(frame_count),
#         disable=not True,
#         file=tqdm_stream,
#         desc="Align frames",
#     ):
#         # Read coordinates and add border
#         pose_list_bordered = []

#         for i in pose_list:
#             pose_list_bordered.append((int(i[idx][0] + crop_size[0]), int(i[idx][1] + crop_size[1])))

#         punkte = []
#         for i in pose_ref_index:
#             coord = [
#                 pose_list_bordered[i][0],
#                 pose_list_bordered[i][1],
#             ]
#             punkte.append(coord)

#         punkte = [punkte]
#         punkte = np.asarray(punkte)

#         # calculate minimal rectangle around snout and tail
#         rect = cv.minAreaRect(punkte)

#         # change size in rect tuple structure to be equal to crop_size
#         lst = list(rect)
#         # lst[0] = center_belly
#         lst[1] = crop_size
#         rect = tuple(lst)

#         # crop image
#         shifted_points = crop_and_flip(
#             rect=rect,
#             points=pose_list_bordered,
#             ref_index=pose_ref_index,
#         )

#         points.append(shifted_points)

#     time_series = np.zeros((len(pose_list) * 2, frame_count))
#     for i in range(frame_count):
#         idx = 0
#         for j in range(len(pose_list)):
#             time_series[idx : idx + 2, i] = points[i][j]
#             idx += 2

#     return time_series


# def crop_and_flip(
#     rect: Tuple,
#     points: List[np.ndarray],
#     ref_index: Tuple[int, int],
# ) -> List[np.ndarray]:
#     """
#     Crop and flip the image based on the given rectangle and points.

#     Parameters
#     ---------
#     rect : Tuple
#         Rectangle coordinates (center, size, theta).
#     points : List[np.ndarray]
#         List of points.
#     ref_index : Tuple[int, int]
#         Reference indices for alignment.

#     Returns
#     -------
#     Tuple[np.ndarray, List[np.ndarray]]
#         Cropped and flipped image, and shifted points.
#     """
#     # Read out rect structures and convert
#     center, size, theta = rect
#     center, size = tuple(map(int, center)), tuple(map(int, size))

#     # Get rotation matrix
#     M = cv.getRotationMatrix2D(center, theta, 1)

#     # shift DLC points
#     x_diff = center[0] - size[0] // 2
#     y_diff = center[1] - size[1] // 2
#     dlc_points_shifted = []
#     for i in points:
#         point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]
#         point[0] -= x_diff
#         point[1] -= y_diff
#         dlc_points_shifted.append(point)

#     # check if flipped correctly, otherwise flip again
#     if dlc_points_shifted[ref_index[1]][0] >= dlc_points_shifted[ref_index[0]][0]:
#         rect = (
#             (size[0] // 2, size[0] // 2),
#             size,
#             180,
#         )  # should second value be size[1]? Is this relevant to the flip? 3/5/24 KKL
#         center, size, theta = rect
#         center, size = tuple(map(int, center)), tuple(map(int, size))

#         # Get rotation matrix
#         M = cv.getRotationMatrix2D(center, theta, 1)

#         # shift DLC points
#         x_diff = center[0] - size[0] // 2
#         y_diff = center[1] - size[1] // 2

#         points = dlc_points_shifted
#         dlc_points_shifted = []

#         for i in points:
#             point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]
#             point[0] -= x_diff
#             point[1] -= y_diff
#             dlc_points_shifted.append(point)

#     return dlc_points_shifted
