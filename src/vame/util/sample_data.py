import time
from pathlib import Path
from typing import Callable, TypeVar

from requests.exceptions import RequestException

from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger

T = TypeVar("T")

GIN_UNREACHABLE_MSG = (
    "Could not reach the movement sample-data server (gin.g-node.org) after {attempts} attempts. "
    "The server is likely slow or temporarily unavailable; please try again later."
)


def _with_retries(func: Callable[[], T], attempts: int = 4, base_delay: float = 5.0) -> T:
    """Call func, retrying on network errors with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except RequestException as exc:
            if attempt == attempts:
                raise ConnectionError(GIN_UNREACHABLE_MSG.format(attempts=attempts)) from exc
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(f"Sample-data download failed (attempt {attempt}/{attempts}), retrying in {delay:.0f}s...")
            time.sleep(delay)


def _import_movement_sample_data():
    # movement fetches its metadata from GIN at import time; a failed import isn't cached, so retrying re-runs it
    import movement.sample_data

    return movement.sample_data


def download_sample_data(source_software: str, with_video: bool = True) -> dict:
    """
    Download sample data.

    Parameters
    ----------
    source_software : str
        Source software used for pose estimation.
    with_video : bool, optional
        If True, the video will be downloaded as well. Defaults to True.

    Returns
    -------
    dict
        Dictionary with the paths to the downloaded sample data.
    """
    movement_sample_data = _with_retries(_import_movement_sample_data)

    download_path = Path("~", ".movement", "data").expanduser().resolve()
    if not download_path.exists():
        download_path.mkdir(parents=True, exist_ok=True)

    dataset_options = {
        "DeepLabCut": "DLC_single-mouse_EPM.predictions.csv",
        "SLEAP": "SLEAP_single-mouse_EPM.predictions.slp",
    }

    info_dict = _with_retries(
        lambda: movement_sample_data.fetch_dataset_paths(
            filename=dataset_options[source_software],
            with_video=with_video,
        )
    )

    video_path = info_dict.get("video")
    if video_path and video_path.stem != info_dict["poses"].stem:
        # rename video file to match pose file (use replace so it works on Windows too)
        video_path = video_path.replace(video_path.parent / (str(info_dict["poses"].stem) + video_path.suffix))

    info_dict["video"] = str(video_path) if video_path is not None else ""
    info_dict["poses"] = str(info_dict["poses"])
    info_dict["frame"] = str(info_dict["frame"])
    info_dict["fps"] = movement_sample_data.metadata[dataset_options[source_software]]["fps"]

    return info_dict
