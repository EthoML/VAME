from pytest import fixture, raises
from pathlib import Path
import shutil
import psutil
import time
from typing import List, Optional, Literal
import numpy as np

import vame
from vame.pipeline import VAMEPipeline
from vame.util.auxiliary import write_config


def init_project(
    project_name: str,
    videos: list,
    poses_estimations: list,
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
    working_directory: str,
    egocentric_data: bool = False,
    centered_reference_keypoint: str = "Nose",
    orientation_reference_keypoint: str = "Tailroot",
    paths_to_pose_nwb_series_data: Optional[List[str]] = None,
):
    config_path, config_values = vame.init_new_project(
        project_name=project_name,
        videos=videos,
        poses_estimations=poses_estimations,
        source_software=source_software,
        working_directory=working_directory,
        video_type=".mp4",
        paths_to_pose_nwb_series_data=paths_to_pose_nwb_series_data,
    )

    # Override config values with test values to speed up tests
    config_values["egocentric_data"] = egocentric_data
    config_values["max_epochs"] = 10
    config_values["batch_size"] = 10
    write_config(config_path, config_values)

    project_data = {
        "project_name": project_name,
        "videos": videos,
        "config_path": config_path,
        "config_data": config_values,
        "centered_reference_keypoint": centered_reference_keypoint,
        "orientation_reference_keypoint": orientation_reference_keypoint,
    }

    return project_data


def cleanup_directory(directory):
    """Helper function to clean up the directory and handle Windows-specific issues."""
    try:
        # Wait a moment to ensure all files are closed
        time.sleep(1)

        # Check for any open file handles and warn about them
        for proc in psutil.process_iter(["open_files"]):
            if any(file.path.startswith(str(directory)) for file in proc.info["open_files"] or []):
                print(f"Process {proc.pid} is holding files in {directory}.")

        # Try to delete the directory
        shutil.rmtree(directory)
    except PermissionError as e:
        print(f"PermissionError during cleanup: {e}. Retrying...")
        # Retry after a short delay
        time.sleep(2)
        try:
            shutil.rmtree(directory)
        except Exception as final_error:
            print(f"Final cleanup failed: {final_error}")


# @fixture(scope="session")
# def setup_project_from_folder():
#     project_name = "test_project_from_folder"
#     videos = [str(Path("./tests/tests_project_sample_data").resolve())]
#     poses_estimations = [str(Path("./tests/tests_project_sample_data").resolve())]
#     working_directory = str(Path("./tests").resolve())

#     # Initialize project
#     project_data = init_project(
#         project_name=project_name,
#         videos=videos,
#         poses_estimations=poses_estimations,
#         source_software="DeepLabCut",
#         working_directory=working_directory,
#         egocentric_data=False,
#     )

#     yield project_data

#     # Clean up
#     config_path = project_data["config_path"]
#     cleanup_directory(Path(config_path).parent)


@fixture(scope="session")
def setup_project_not_aligned_data():
    project_name = "test_project_align"
    videos = [str(Path("./tests/tests_project_sample_data/cropped_video.mp4").resolve())]
    poses_estimations = [str(Path("./tests/tests_project_sample_data/cropped_video.csv").resolve())]
    working_directory = str(Path("./tests").resolve())

    # Initialize project
    project_data = init_project(
        project_name=project_name,
        videos=videos,
        poses_estimations=poses_estimations,
        source_software="DeepLabCut",
        working_directory=working_directory,
        egocentric_data=False,
    )

    yield project_data

    # Clean up
    config_path = project_data["config_path"]
    cleanup_directory(Path(config_path).parent)


# # TODO change to test fixed (already egocentrically aligned) data when have it
@fixture(scope="session")
def setup_project_fixed_data():
    project_name = "test_project_fixed"
    videos = [str(Path("./tests/tests_project_sample_data/cropped_video.mp4").resolve())]
    poses_estimations = [str(Path("./tests/tests_project_sample_data/cropped_video.csv").resolve())]
    working_directory = str(Path("./tests").resolve())

    # Initialize project
    project_data = init_project(
        project_name=project_name,
        videos=videos,
        poses_estimations=poses_estimations,
        source_software="DeepLabCut",
        working_directory=working_directory,
        egocentric_data=True,
    )

    yield project_data

    # Clean up
    config_path = project_data["config_path"]
    cleanup_directory(Path(config_path).parent)


# @fixture(scope="session")
# def setup_nwb_data_project():
#     project_name = "test_project_nwb"
#     videos = ["./tests/tests_project_sample_data/cropped_video.mp4"]
#     poses_estimations = ["./tests/test_project_sample_nwb/cropped_video.nwb"]
#     paths_to_pose_nwb_series_data = [
#         "processing/behavior/data_interfaces/PoseEstimation/pose_estimation_series"
#     ]
#     working_directory = "./tests"

#     # Initialize project
#     config, project_data = init_project(
#         project_name=project_name,
#         videos=videos,
#         poses_estimations=poses_estimations,
#         working_directory=working_directory,
#         egocentric_data=False,
#         paths_to_pose_nwb_series_data=paths_to_pose_nwb_series_data,
#     )

#     yield project_data

#     # Clean up
#     shutil.rmtree(Path(config).parent)


@fixture(scope="session")
def setup_project_and_convert_pose_to_numpy(setup_project_fixed_data):
    config = setup_project_fixed_data["config_data"]
    vame.pose_to_numpy(config, save_logs=True)
    return setup_project_fixed_data


@fixture(scope="session")
def setup_project_and_align_egocentric(setup_project_not_aligned_data):
    config_data = setup_project_not_aligned_data["config_data"]
    centered_reference_keypoint = setup_project_not_aligned_data["centered_reference_keypoint"]
    orientation_reference_keypoint = setup_project_not_aligned_data["orientation_reference_keypoint"]
    vame.preprocessing(
        config=config_data,
        centered_reference_keypoint=centered_reference_keypoint,
        orientation_reference_keypoint=orientation_reference_keypoint,
        save_logs=True,
    )
    return setup_project_not_aligned_data


@fixture(scope="function")
def setup_project_and_check_param_aligned_dataset(setup_project_and_align_egocentric):
    config = setup_project_and_align_egocentric["config_data"]
    vame.create_trainset(
        config=config,
        save_logs=True,
    )
    return setup_project_and_align_egocentric


@fixture(scope="function")
def setup_project_and_check_param_fixed_dataset(
    setup_project_and_convert_pose_to_numpy,
):
    # use setup_project_and_align_egocentric fixture or setup_project_and_convert_pose_to_numpy based on value of egocentric_aligned
    config = setup_project_and_convert_pose_to_numpy["config_data"]
    with raises(NotImplementedError, match="Fixed data training is not implemented yet"):
        vame.create_trainset(
            config=config,
            save_logs=True,
        )
    return setup_project_and_convert_pose_to_numpy


@fixture(scope="session")
def setup_project_and_create_train_aligned_dataset(setup_project_and_align_egocentric):
    config = setup_project_and_align_egocentric["config_data"]
    vame.create_trainset(
        config=config,
        split_mode="mode_2",
        save_logs=False,
    )
    return setup_project_and_align_egocentric


# @fixture(scope="session")
# def setup_project_and_create_train_fixed_dataset(
#     setup_project_and_convert_pose_to_numpy,
# ):
#     # use setup_project_and_align_egocentric fixture or setup_project_and_convert_pose_to_numpy based on value of egocentric_aligned
#     config = setup_project_and_convert_pose_to_numpy["config_data"]
#     vame.create_trainset(
#         config=config,
#         save_logs=True,
#     )
#     return setup_project_and_convert_pose_to_numpy


@fixture(scope="session")
def setup_project_and_train_model(setup_project_and_create_train_aligned_dataset):
    config = setup_project_and_create_train_aligned_dataset["config_data"]
    vame.train_model(config, save_logs=True)
    return setup_project_and_create_train_aligned_dataset


@fixture(scope="session")
def setup_project_and_evaluate_model(setup_project_and_train_model):
    config = setup_project_and_train_model["config_data"]
    vame.evaluate_model(config, save_logs=True)
    return setup_project_and_train_model


@fixture(scope="session")
def setup_pipeline():
    """
    Setup a Pipeline for testing.
    """
    project_name = "test_pipeline"
    videos = ["./tests/tests_project_sample_data/cropped_video.mp4"]
    poses_estimations = ["./tests/tests_project_sample_data/cropped_video.csv"]
    working_directory = str(Path("./tests").resolve())
    source_software = "DeepLabCut"

    config_kwargs = {
        "egocentric_data": False,
        "max_epochs": 10,
        "batch_size": 10,
    }
    pipeline = VAMEPipeline(
        working_directory=working_directory,
        project_name=project_name,
        videos=videos,
        poses_estimations=poses_estimations,
        source_software=source_software,
        config_kwargs=config_kwargs,
    )
    yield {"pipeline": pipeline}

    # Clean up
    cleanup_directory(Path(pipeline.config_path).parent)
