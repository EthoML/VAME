import numpy as np
from typing import List, Tuple, Optional
import cv2 as cv
import os
from scipy.ndimage import median_filter
import tqdm
from pynwb import NWBHDF5IO
from pynwb.file import NWBFile
from hdmf.utils import LabelledDict
import pandas as pd

from vame.schemas.project import PoseEstimationFiletype
from vame.logging.logger import VameLogger
from vame.io.load_poses import load_vame_dataset


logger_config = VameLogger(__name__)
logger = logger_config.logger


def get_pose_data_from_nwb_file(
    nwbfile: NWBFile,
    path_to_pose_nwb_series_data: str,
) -> LabelledDict:
    """
    Get pose data from nwb file using a inside path to the nwb data.

    Parameters:
    ----------
    nwbfile : NWBFile)
        NWB file object.
    path_to_pose_nwb_series_data : str
        Path to the pose data inside the nwb file.

    Returns
    -------
    LabelledDict
        Pose data.
    """
    if not path_to_pose_nwb_series_data:
        raise ValueError("Path to pose nwb series data is required.")
    pose_data = nwbfile
    for key in path_to_pose_nwb_series_data.split("/"):
        if isinstance(pose_data, dict):
            pose_data = pose_data.get(key)
            continue
        pose_data = getattr(pose_data, key)
    return pose_data


def get_dataframe_from_pose_nwb_file(
    file_path: str,
    path_to_pose_nwb_series_data: str,
) -> pd.DataFrame:
    """
    Get pose data from nwb file and return it as a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the nwb file.
    path_to_pose_nwb_series_data : str
        Path to the pose data inside the nwb file.

    Returns
    -------
    pd.DataFrame
        Pose data as a pandas DataFrame.
    """
    with NWBHDF5IO(file_path, "r") as io:
        nwbfile = io.read()
        pose = get_pose_data_from_nwb_file(nwbfile, path_to_pose_nwb_series_data)
        dataframes = []
        for label, pose_series in pose.items():
            data = pose_series.data[:]
            confidence = pose_series.confidence[:]
            df = pd.DataFrame(data, columns=[f"{label}_x", f"{label}_y"])
            df[f"likelihood_{label}"] = confidence
            dataframes.append(df)
        final_df = pd.concat(dataframes, axis=1)
    return final_df


def read_pose_estimation_file(
    file_path: str,
    file_type: Optional[PoseEstimationFiletype] = None,
    path_to_pose_nwb_series_data: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Read pose estimation file.

    Parameters
    ----------
    file_path : str
        Path to the pose estimation file.
    file_type : PoseEstimationFiletype
        Type of the pose estimation file. Supported types are 'csv' and 'nwb'.
    path_to_pose_nwb_series_data : str, optional
        Path to the pose data inside the nwb file, by default None

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        Tuple containing the pose estimation data as a pandas DataFrame and a numpy array.
    """
    ds = load_vame_dataset(ds_path=file_path)
    data = nc_to_dataframe(ds)
    data_mat = pd.DataFrame.to_numpy(data)
    return data, data_mat
    # if file_type == PoseEstimationFiletype.csv:
    #     data = pd.read_csv(file_path, skiprows=2, index_col=0)
    #     if "coords" in data:
    #         data = data.drop(columns=["coords"], axis=1)
    #     data_mat = pd.DataFrame.to_numpy(data)
    #     return data, data_mat
    # elif file_type == PoseEstimationFiletype.nwb:
    #     if not path_to_pose_nwb_series_data:
    #         raise ValueError("Path to pose nwb series data is required.")
    #     data = get_dataframe_from_pose_nwb_file(
    #         file_path=file_path,
    #         path_to_pose_nwb_series_data=path_to_pose_nwb_series_data,
    #     )
    #     data_mat = pd.DataFrame.to_numpy(data)
    #     return data, data_mat
    # raise ValueError(f"Filetype {file_type} not supported")


def consecutive(
    data: np.ndarray,
    stepsize: int = 1,
) -> List[np.ndarray]:
    """
    Find consecutive sequences in the data array.

    Parameters
    ----------
    data : np.ndarray
        Input array.
    stepsize : int, optional
        Step size. Defaults to 1.

    Returns
    -------
    List[np.ndarray]
        List of consecutive sequences.
    """
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def nan_helper(y: np.ndarray) -> Tuple:
    """
    Identifies indices of NaN values in an array and provides a function to convert them to non-NaN indices.

    Parameters
    ----------
    y : np.ndarray
        Input array containing NaN values.

    Returns
    -------
    Tuple[np.ndarray, Union[np.ndarray, None]]
        A tuple containing two elements:
        - An array of boolean values indicating the positions of NaN values.
        - A lambda function to convert NaN indices to non-NaN indices.
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpol_all_nans(arr: np.ndarray) -> np.ndarray:
    """
    Interpolates all NaN values in the given array.

    Parameters
    ----------
    arr : np.ndarray
        Input array containing NaN values.

    Returns
    -------
    np.ndarray
        Array with NaN values replaced by interpolated values.
    """
    y = np.transpose(arr)
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    arr = np.transpose(y)
    return arr


def interpol_first_rows_nans(arr: np.ndarray) -> np.ndarray:
    """
    Interpolates NaN values in the given array.

    Parameters
    ----------
    arr : np.ndarray
        Input array with NaN values.

    Returns
    -------
    np.ndarray
        Array with interpolated NaN values.
    """
    y = np.transpose(arr)
    nans, x = nan_helper(y[0])
    y[0][nans] = np.interp(x(nans), x(~nans), y[0][~nans])
    nans, x = nan_helper(y[1])
    y[1][nans] = np.interp(x(nans), x(~nans), y[1][~nans])
    arr = np.transpose(y)
    return arr


def crop_and_flip(
    rect: Tuple,
    src: np.ndarray,
    points: List[np.ndarray],
    ref_index: Tuple[int, int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Crop and flip the image based on the given rectangle and points.

    Parameters
    ----------
    rect : Tuple
        Rectangle coordinates (center, size, theta).
    src: np.ndarray
        Source image.
    points : List[np.ndarray]
        List of points.
    ref_index : Tuple[int, int]
        Reference indices for alignment.

    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        Cropped and flipped image, and shifted points.
    """
    # Read out rect structures and convert
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # Get rotation matrix
    M = cv.getRotationMatrix2D(center, theta, 1)

    # shift DLC points
    x_diff = center[0] - size[0] // 2
    y_diff = center[1] - size[1] // 2
    dlc_points_shifted = []
    for i in points:
        point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]
        point[0] -= x_diff
        point[1] -= y_diff
        dlc_points_shifted.append(point)

    # Perform rotation on src image
    dst = cv.warpAffine(src.astype("float32"), M, src.shape[:2])
    out = cv.getRectSubPix(dst, size, center)

    # check if flipped correctly, otherwise flip again
    if dlc_points_shifted[ref_index[1]][0] >= dlc_points_shifted[ref_index[0]][0]:
        rect = (
            (size[0] // 2, size[0] // 2),
            size,
            180,
        )  # should second value be size[1]? Is this relevant to the flip? 3/5/24 KKL
        center, size, theta = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # Get rotation matrix
        M = cv.getRotationMatrix2D(center, theta, 1)

        # shift DLC points
        x_diff = center[0] - size[0] // 2
        y_diff = center[1] - size[1] // 2

        points = dlc_points_shifted
        dlc_points_shifted = []

        for i in points:
            point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]
            point[0] -= x_diff
            point[1] -= y_diff
            dlc_points_shifted.append(point)

        # Perform rotation on src image
        dst = cv.warpAffine(out.astype("float32"), M, out.shape[:2])
        out = cv.getRectSubPix(dst, size, center)
    return out, dlc_points_shifted


def background(
    project_path: str,
    session: str,
    video_path: str,
    num_frames: int = 1000,
    save_background: bool = True,
) -> np.ndarray:
    """
    Compute background image from fixed camera.

    Parameters
    ----------
    project_path : str
        Path to the project directory.
    session : str
        Name of the session.
    video_path : str
        Path to the video file.
    num_frames : int, optional
        Number of frames to use for background computation. Defaults to 1000.

    Returns
    -------
    np.ndarray
        Background image.
    """
    logger.info("Computing background image ...")

    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        raise Exception(f"Unable to open video file: {video_path}")

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()
    height, width, _ = frame.shape
    frames = np.zeros((height, width, num_frames))

    for i in tqdm.tqdm(
        range(num_frames),
        disable=not True,
        desc="Compute background image for session %s" % session,
    ):
        rand = np.random.choice(frame_count, replace=False)
        capture.set(1, rand)
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[..., i] = gray

    logger.info("Finishing up!")
    medFrame = np.median(frames, 2)
    background = median_filter(medFrame, (5, 5))

    if save_background:
        np.save(
            os.path.join(
                project_path,
                "data",
                "processed",
                session + "-background.npy",
            ),
            background,
        )

    capture.release()
    return background


def nc_to_dataframe(nc_data):
    keypoints = nc_data["keypoints"].values
    space = nc_data["space"].values

    # Flatten position data
    position_data = nc_data["position"].isel(individuals=0).values
    position_column_names = [
        f"{keypoint}_{sp}" for keypoint in keypoints for sp in space
    ]
    position_flattened = position_data.reshape(position_data.shape[0], -1)

    # Create a DataFrame for position data
    position_df = pd.DataFrame(position_flattened, columns=position_column_names)

    # Extract and flatten confidence data
    confidence_data = nc_data["confidence"].isel(individuals=0).values
    confidence_column_names = [f"{keypoint}_confidence" for keypoint in keypoints]
    confidence_flattened = confidence_data.reshape(confidence_data.shape[0], -1)
    confidence_df = pd.DataFrame(confidence_flattened, columns=confidence_column_names)

    # Combine position and confidence data
    combined_df = pd.concat([position_df, confidence_df], axis=1)

    # Reorder columns: keypoint_x, keypoint_y, keypoint_confidence
    reordered_columns = []
    for keypoint in keypoints:
        reordered_columns.extend(
            [f"{keypoint}_x", f"{keypoint}_y", f"{keypoint}_confidence"]
        )

    combined_df = combined_df[reordered_columns]

    return combined_df
