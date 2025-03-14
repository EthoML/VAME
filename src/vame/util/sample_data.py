from movement.sample_data import fetch_dataset_paths
from pathlib import Path


DOWNLOAD_PATH = Path("~", ".movement", "data").expanduser().resolve()
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

dataset_options = {
    "DeepLabCut": "DLC_single-mouse_EPM.predictions.csv",
    "SLEAP": "SLEAP_single-mouse_EPM.predictions.slp",
}


def download_sample_data(source_software: str) -> dict:
    """
    Download sample data.

    Parameters
    ----------
    source_software : str
        Source software used for pose estimation.

    Returns
    -------
    dict
        Dictionary with the paths to the downloaded sample data.
    """
    paths_dict = fetch_dataset_paths(
        filename=dataset_options[source_software],
        with_video=True,
    )

    video_path = paths_dict.get("video")
    if video_path and video_path.stem != paths_dict["poses"].stem:
        # rename video file to match pose file
        video_path = video_path.rename(
            video_path.parent / (str(paths_dict["poses"].stem) + video_path.suffix)
        )

    paths_dict["video"] = str(video_path)
    paths_dict["poses"] = str(paths_dict["poses"])
    paths_dict["frame"] = str(paths_dict["frame"])

    return paths_dict
