import cv2
from typing import List
from pathlib import Path
import shutil
import os

from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger


def get_video_frame_rate(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise Exception(f"Unable to open video file: {video_path}")
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    return frame_rate


def add_videos_to_project(
    config: dict,
    videos: List[str],
    copy_videos: bool = False,
) -> None:
    """
    Add videos to the project, ensuring video stems match session names.

    Parameters
    ----------
    config : dict
        Configuration settings for the project.
    videos : List[str]
        List of video file paths to add.
    copy_videos : bool, optional
        If True, copy videos to project directory. If False, create symbolic links, by default False

    Raises
    ------
    ValueError
        If video stems don't match session names.
    """
    session_names = config.get("session_names", [])
    if not session_names:
        raise ValueError("No session names found in config")

    if len(videos) != len(session_names):
        raise ValueError(
            f"Number of videos ({len(videos)}) does not match number of session names ({len(session_names)})"
        )

    # Extract stems from video paths and verify they match session names
    video_stems = [Path(video).stem for video in videos]
    for video_stem in video_stems:
        if video_stem not in session_names:
            raise ValueError(
                f"Video stem '{video_stem}' does not match any session name in {session_names}"
            )

    # Copy / link videos to project directory
    project_path = Path(config["project_path"])
    data_raw_path = project_path / "data" / "raw"
    destinations = [data_raw_path / Path(vp).name for vp in videos]
    for src, dst in zip(videos, destinations):
        if copy_videos:
            logger.info(f"Copying {src} to {dst}")
            shutil.copy(os.fspath(src), os.fspath(dst))
        else:
            logger.info(f"Creating symbolic link from {src} to {dst}")
            os.symlink(os.fspath(src), os.fspath(dst))


# def play_aligned_video(
#     a: List[np.ndarray],
#     n: List[List[np.ndarray]],
#     frame_count: int,
# ) -> None:
#     """
#     Play the aligned video.

#     Parameters
#     ---------
#     a : List[np.ndarray]
#         List of aligned images.
#     n : List[List[np.ndarray]]
#         List of aligned DLC points.
#     frame_count : int
#         Number of frames in the video.
#     """
#     colors = [
#         (255, 0, 0),
#         (0, 255, 0),
#         (0, 0, 255),
#         (255, 255, 0),
#         (255, 0, 255),
#         (0, 255, 255),
#         (0, 0, 0),
#         (255, 255, 255),
#     ]
#     for i in range(frame_count):
#         # Capture frame-by-frame
#         ret, frame = True, a[i]
#         if ret is True:
#             # Display the resulting frame
#             frame = cv2.cvtColor(frame.astype("uint8") * 255, cv2.COLOR_GRAY2BGR)
#             im_color = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
#             for c, j in enumerate(n[i]):
#                 cv2.circle(im_color, (j[0], j[1]), 5, colors[c], -1)
#             cv2.imshow("Frame", im_color)
#             # Press Q on keyboard to exit
#             # Break the loop
#             if cv2.waitKey(25) & 0xFF == ord("q"):
#                 break
#         else:
#             break
#     cv2.destroyAllWindows()
