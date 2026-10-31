import time
from pathlib import Path
from typing import Callable, TypeVar

import pooch
import yaml
from requests.exceptions import RequestException

from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger

T = TypeVar("T")

SAMPLE_DATA_URL = "https://gin.swc.ucl.ac.uk/neuroinformatics/movement-sample-data/raw/master"
# Same cache folder and layout as movement, so earlier downloads are reused
DOWNLOAD_PATH = Path("~", ".movement", "data").expanduser()

DATASETS = {
    "DeepLabCut": "DLC_single-mouse_EPM.predictions.csv",
    "SLEAP": "SLEAP_single-mouse_EPM.predictions.slp",
}

GIN_UNREACHABLE_MSG = (
    "Could not reach the movement sample-data server (gin.swc.ucl.ac.uk) after {attempts} attempts. "
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


def _fetch(remote_path: str, sha256: str | None, fname: str | None = None) -> Path:
    """Download one file from the sample-data repository, or reuse the cached copy if its hash matches."""
    remote = Path(remote_path)
    return Path(
        _with_retries(
            lambda: pooch.retrieve(
                url=f"{SAMPLE_DATA_URL}/{remote_path}",
                known_hash=f"sha256:{sha256}" if sha256 else None,
                fname=fname or remote.name,
                path=DOWNLOAD_PATH / remote.parent,
                progressbar=True,
            )
        )
    )


def download_sample_data(source_software: str, with_video: bool = True) -> dict:
    """
    Download sample data.

    Files are downloaded from movement's sample-data repository on SWC GIN and cached in
    ~/.movement/data. This bypasses `movement.sample_data` for now: up to movement 0.17.0 it
    downloads from G-Node GIN (gin.g-node.org), which is often unreachable. movement moved to
    SWC GIN in https://github.com/neuroinformatics-unit/movement/pull/1080, which is not yet
    released. Once a movement release includes it, go back to
    `movement.sample_data.fetch_dataset_paths` and require that version.

    Parameters
    ----------
    source_software : str
        Source software used for pose estimation.
    with_video : bool, optional
        If True, the video will be downloaded as well. Defaults to True.

    Returns
    -------
    dict
        Dictionary with the paths to the downloaded sample data ("poses", "video", "frame")
        and the video frame rate ("fps"). The video is saved under the pose file's name.
    """
    with open(_fetch("metadata.yaml", sha256=None)) as f:
        metadata = yaml.safe_load(f)

    filename = DATASETS[source_software]
    entry = metadata[filename]

    poses = _fetch(f"poses/{filename}", entry["sha256sum"])
    frame = _fetch(f"frames/{entry['frame']['file_name']}", entry["frame"]["sha256sum"])
    video = ""
    if with_video:
        video_file = entry["video"]["file_name"]
        # Saved under the pose file's name, as before, so pose and video share a session name
        video = _fetch(
            f"videos/{video_file}",
            entry["video"]["sha256sum"],
            fname=f"{poses.stem}{Path(video_file).suffix}",
        )

    return {
        "poses": str(poses),
        "video": str(video),
        "frame": str(frame),
        "fps": entry["fps"],
    }
