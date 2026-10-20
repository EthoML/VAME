from pathlib import Path
from pytest import raises
import os
import matplotlib
import pytest

from vame.visualization import (
    visualize_preprocessing_scatter,
    visualize_preprocessing_timeseries,
    visualize_preprocessing_cloud,
)


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
    assert Path(paths_dict["video"]).stem == Path(paths_dict["poses"]).stem
    assert paths_dict["fps"] == 30


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


@pytest.mark.parametrize(
    "plot_function",
    [visualize_preprocessing_scatter, visualize_preprocessing_timeseries, visualize_preprocessing_cloud],
)
def test_preprocessing_plots_keep_matplotlib_backend(setup_project_and_align_egocentric, plot_function):
    """Plotting with show_figure=False does not change the session's matplotlib backend"""
    config = setup_project_and_align_egocentric["config_data"]
    matplotlib.use("pdf")
    try:
        plot_function(config=config, show_figure=False)
        assert matplotlib.get_backend() == "pdf"
    finally:
        matplotlib.use("Agg")


def test_save_state_records_positional_args_and_function_defaults(tmp_path):
    """save_state binds args to the function signature and records its defaults"""
    import json
    from vame.schemas.states import save_state, CreateTrainsetFunctionSchema, VAMEPipelineStatesSchema

    (tmp_path / "states").mkdir()
    states_file = tmp_path / "states" / "states.json"
    states_file.write_text(json.dumps(VAMEPipelineStatesSchema().model_dump()))

    @save_state(model=CreateTrainsetFunctionSchema)
    def create_trainset(config, test_fraction=0.1, split_mode="mode_2", keypoints_to_include=None):
        return test_fraction

    assert create_trainset({"project_path": str(tmp_path)}, 0.2, keypoints_to_include=["nose"]) == 0.2
    record = json.loads(states_file.read_text())["create_trainset"]
    assert record == {"execution_state": "success", "test_fraction": 0.2, "split_mode": "mode_2"}
