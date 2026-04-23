from pathlib import Path
from vame.util.auxiliary import read_config
from vame import init_new_project
from vame.io.load_poses import load_vame_dataset
import shutil


def test_project_config_exists(setup_project_not_aligned_data):
    """
    Test if the project config file exists.
    """
    assert Path(setup_project_not_aligned_data["config_path"]).exists()


def test_project_name_config(setup_project_not_aligned_data):
    """
    Test if the project name is correctly set in the config file.
    """
    config = Path(setup_project_not_aligned_data["config_path"])
    config_values = read_config(config)
    assert config_values["project_name"] == setup_project_not_aligned_data["project_name"]


def test_existing_project():
    project_name = "test_existing_project"
    videos = ["./tests/tests_project_sample_data/cropped_video.mp4"]
    poses_estimations = ["./tests/tests_project_sample_data/cropped_video.csv"]
    working_directory = "./tests"

    config_path_creation, config_creation = init_new_project(
        project_name=project_name,
        videos=videos,
        poses_estimations=poses_estimations,
        source_software="DeepLabCut",
        working_directory=working_directory,
        copy_videos=True,
    )
    config_path_duplicated, config_duplicated = init_new_project(
        project_name=project_name,
        videos=videos,
        poses_estimations=poses_estimations,
        source_software="DeepLabCut",
        working_directory=working_directory,
        copy_videos=True,
    )
    assert config_path_creation == config_path_duplicated
    shutil.rmtree(Path(config_path_creation).parent)


# def test_existing_project_from_folder(setup_project_from_folder):
#     config = Path(setup_project_from_folder["config_path"])
#     config_values = read_config(config)
#     assert config_values["project_name"] == setup_project_from_folder["project_name"]
#     assert Path(setup_project_from_folder["config_path"]).exists()


def test_init_project_from_nwb():
    """
    Initialize a project from an NWB file (ndx-pose) and verify that the
    session is registered and the keypoints from the NWB PoseEstimation
    container are preserved in the generated xarray dataset.
    """
    project_name = "test_project_nwb"
    nwb_path = Path("./tests/tests_project_sample_nwb/cropped_video.nwb").resolve()
    working_directory = str(Path("./tests").resolve())

    config_path, config_values = init_new_project(
        project_name=project_name,
        poses_estimations=[str(nwb_path)],
        source_software="NWB",
        working_directory=working_directory,
    )

    try:
        assert Path(config_path).exists()
        assert config_values["session_names"] == [nwb_path.stem]
        assert config_values["pose_estimation_filetype"] == "nwb"

        ds_path = Path(config_values["project_path"]) / "data" / "raw" / f"{nwb_path.stem}.nc"
        ds = load_vame_dataset(ds_path)
        expected_keypoints = {
            "Forehand-Left",
            "Forehand-Right",
            "Hindhand-Left",
            "Hindhand-Right",
            "Nose",
            "Tailroot",
        }
        assert set(ds["keypoints"].values.tolist()) == expected_keypoints
    finally:
        shutil.rmtree(Path(config_path).parent)
