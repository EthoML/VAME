from pathlib import Path


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
    from movement.sample_data import fetch_dataset_paths

    download_path = Path("~", ".movement", "data").expanduser().resolve()
    if not download_path.exists():
        download_path.mkdir(parents=True, exist_ok=True)

    dataset_options = {
        "DeepLabCut": "DLC_single-mouse_EPM.predictions.csv",
        "SLEAP": "SLEAP_single-mouse_EPM.predictions.slp",
    }

    paths_dict = fetch_dataset_paths(
        filename=dataset_options[source_software],
        with_video=with_video,
    )

    video_path = paths_dict.get("video")
    if video_path and video_path.stem != paths_dict["poses"].stem:
        # rename video file to match pose file
        video_path = video_path.rename(video_path.parent / (str(paths_dict["poses"].stem) + video_path.suffix))

    paths_dict["video"] = str(video_path) if video_path is not None else ""
    paths_dict["poses"] = str(paths_dict["poses"])
    paths_dict["frame"] = str(paths_dict["frame"])

    return paths_dict
