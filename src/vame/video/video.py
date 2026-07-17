import cv2
from typing import List, Optional, Union
from pathlib import Path
import shutil
import os

from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger


# Container formats accepted as project input
VIDEO_SUFFIXES = (".mp4", ".avi")


def is_video_file(path: Union[str, Path]) -> bool:
    """Whether ``path`` looks like a video VAME can read, by extension."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def resolve_video_type(videos: List[Union[str, Path]]) -> Optional[str]:
    """
    The extension shared by ``videos``, or None when they are not all the same.

    Recorded in the config as the project's video format. A mixed-format project
    has no single answer, so callers fall back to resolving each session's video
    from disk.
    """
    suffixes = {Path(v).suffix.lower() for v in videos if v}
    if len(suffixes) == 1:
        return suffixes.pop()
    if len(suffixes) > 1:
        logger.info(f"Project has mixed video formats ({', '.join(sorted(suffixes))}); leaving 'video_type' unset.")
    return None


def get_session_video_path(
    config: dict,
    session: str,
) -> Path:
    """
    Resolve the raw video belonging to ``session``.

    Videos are linked into ``data/raw`` under their session name, keeping their
    original extension, so the format is read back off disk rather than assumed.
    ``video_type`` in the config only breaks ties when a session has more than
    one video file.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    session : str
        Name of the session.

    Returns
    -------
    Path
        Path to the session's video file.

    Raises
    ------
    FileNotFoundError
        If the session has no video file in ``data/raw``.
    """
    data_raw_path = Path(config["project_path"]) / "data" / "raw"
    # Matched on the exact stem rather than a glob, so session names containing
    # glob metacharacters (and prefixes like Session1 vs Session10) stay distinct.
    candidates = sorted(
        p for p in (data_raw_path.iterdir() if data_raw_path.is_dir() else []) if p.stem == session and is_video_file(p)
    )

    if not candidates:
        raise FileNotFoundError(
            f"No video file found for session '{session}' in {data_raw_path}. "
            f"Expected a file named '{session}' with one of: {', '.join(VIDEO_SUFFIXES)}. "
            "Videos are linked into the project when it is created; if this project was "
            "created without videos, add them with vame.video.add_videos_to_project()."
        )

    if len(candidates) > 1:
        preferred = str(config.get("video_type", "")).lower()
        for candidate in candidates:
            if candidate.suffix.lower() == preferred:
                return candidate
        logger.info(
            f"Session '{session}' has multiple videos ({', '.join(c.name for c in candidates)}); "
            f"using {candidates[0].name}. Set 'video_type' in the config to pick another."
        )

    return candidates[0]


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
    # Imported lazily to keep this module's import graph free of torch.
    from vame.util.auxiliary import write_config

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
            raise ValueError(f"Video stem '{video_stem}' does not match any session name in {session_names}")

    for video in videos:
        if not is_video_file(video):
            raise ValueError(
                f"Unsupported video format '{Path(video).suffix}' for {video}. "
                f"Must be one of: {', '.join(VIDEO_SUFFIXES)}."
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
            try:
                logger.info(f"Creating symbolic link from {src} to {dst}")
                os.symlink(os.fspath(src), os.fspath(dst))
            except OSError as e:
                raise OSError(
                    f"Failed to create a symbolic link from {src} to {dst}. "
                    "On Windows, symlinks require Administrator privileges or Developer Mode. "
                    "Enable Developer Mode in Windows Settings, run as Administrator, "
                    "or pass copy_videos=True to copy the files instead."
                ) from e

    config["video_type"] = resolve_video_type(videos)
    write_config(
        config_path=str(project_path / "config.yaml"),
        config=config,
    )


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
