from pathlib import Path
from pytest import raises
import os


def test_pose_to_numpy_file_exists(setup_project_and_convert_pose_to_numpy):
    """
    Test if the pose-estimation file was converted to a numpy array file.
    """
    project_path = setup_project_and_convert_pose_to_numpy["config_data"]["project_path"]
    file_name = setup_project_and_convert_pose_to_numpy["config_data"]["session_names"][0]
    file_path = os.path.join(
        project_path,
        "data",
        "processed",
        f"{file_name}-PE-seq.npy",
    )
    assert os.path.exists(file_path)


def test_egocentric_alignment_file_is_created(setup_project_and_align_egocentric):
    """
    Test if the egocentric alignment function creates the expected file.
    """
    project_path = setup_project_and_align_egocentric["config_data"]["project_path"]
    session_name = setup_project_and_align_egocentric["config_data"]["session_names"][0]
    file_path = os.path.join(
        project_path,
        "data",
        "processed",
        f"{session_name}_processed.nc",
    )
    assert os.path.exists(file_path)


# def test_nwb_data_egocentric_alignment_file_is_created(setup_nwb_data_project):
#     """
#     Test if the egocentric alignment function creates the expected file using NWB data.
#     """
#     config_path = setup_nwb_data_project["config_path"]
#     vame.egocentric_alignment(
#         config_path,
#         pose_ref_index=setup_nwb_data_project["pose_ref_index"],
#         save_logs=True,
#     )
#     project_path = setup_nwb_data_project["config_data"]["project_path"]
#     file_name = setup_nwb_data_project["config_data"]["session_names"][0]
#     file_path = os.path.join(
#         project_path, "data", "processed", file_name, f"{file_name}-PE-seq.npy"
#     )
#     assert os.path.exists(file_path)


def test_download_sample_data():
    """
    Test if the download_sample_data function returns a dictionary with the expected keys.
    """
    from vame.util.sample_data import download_sample_data

    source_software = "DeepLabCut"
    paths_dict = download_sample_data(source_software)
    assert "video" in paths_dict
    assert Path(paths_dict["video"]).exists()
    assert "poses" in paths_dict
    assert Path(paths_dict["poses"]).exists()
    assert "frame" in paths_dict
    assert Path(paths_dict["frame"]).exists()


def _raw_project(tmp_path, filenames):
    """A bare project dir whose data/raw holds `filenames`."""
    data_raw = tmp_path / "data" / "raw"
    data_raw.mkdir(parents=True)
    for name in filenames:
        (data_raw / name).touch()
    return {"project_path": str(tmp_path)}


def test_get_session_video_path_resolves_non_mp4_source(tmp_path):
    """
    A session's video is found by whatever extension it was loaded with.
    Regression test for AVI projects failing at video creation.
    """
    from vame.video.video import get_session_video_path

    config = _raw_project(tmp_path, ["Session1.avi", "Session1.nc"])
    resolved = get_session_video_path(config=config, session="Session1")
    assert resolved.name == "Session1.avi"


def test_get_session_video_path_ignores_pose_files(tmp_path):
    """
    Pose data lives in data/raw next to the videos and must never be handed to OpenCV.
    """
    from vame.video.video import get_session_video_path

    config = _raw_project(tmp_path, ["Session1.nc"])
    with raises(FileNotFoundError):
        get_session_video_path(config=config, session="Session1")


def test_get_session_video_path_does_not_match_other_sessions(tmp_path):
    """
    A session prefix must not pull in a different session's video.
    """
    from vame.video.video import get_session_video_path

    config = _raw_project(tmp_path, ["Session1.mp4", "Session10.avi"])
    assert get_session_video_path(config=config, session="Session10").name == "Session10.avi"
    assert get_session_video_path(config=config, session="Session1").name == "Session1.mp4"
