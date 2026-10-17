import shutil
from pathlib import Path

import pytest
import xarray as xr

import vame
from vame.io.load_poses import load_vame_dataset
from vame.preprocessing.alignment import egocentrically_align_and_center
from vame.preprocessing.cleaning import lowconf_cleaning, outlier_cleaning
from vame.preprocessing.filter import savgol_filtering
from vame.preprocessing.scaling import rescaling

SAMPLE_POSES = Path(__file__).parent / "tests_project_sample_data" / "cropped_video.csv"
CENTER = "Nose"
ORIENTATION = "Tailroot"


@pytest.fixture(scope="module")
def pose_files(tmp_path_factory):
    poses_dir = tmp_path_factory.mktemp("poses")
    files = []
    for name in ("session_a", "session_b", "session_c"):
        dst = poses_dir / f"{name}.csv"
        shutil.copy(SAMPLE_POSES, dst)
        files.append(str(dst))
    return files


def make_project(working_directory, project_name, pose_files):
    _, config = vame.init_new_project(
        project_name=project_name,
        poses_estimations=pose_files,
        source_software="DeepLabCut",
        working_directory=str(working_directory),
    )
    return config


def processed_datasets(config):
    processed_dir = Path(config["project_path"]) / "data" / "processed"
    return {s: load_vame_dataset(processed_dir / f"{s}_processed.nc") for s in config["session_names"]}


def assert_same_processed(config_a, config_b):
    datasets_a = processed_datasets(config_a)
    datasets_b = processed_datasets(config_b)
    assert datasets_a.keys() == datasets_b.keys()
    for session in datasets_a:
        xr.testing.assert_identical(datasets_a[session], datasets_b[session])


def test_matches_step_functions(tmp_path, pose_files):
    stepwise = make_project(tmp_path, "stepwise", pose_files)
    combined = make_project(tmp_path, "combined", pose_files)

    lowconf_cleaning(stepwise, read_from_variable="position", save_to_variable="position_cleaned_lowconf")
    egocentrically_align_and_center(
        stepwise,
        centered_reference_keypoint=CENTER,
        orientation_reference_keypoint=ORIENTATION,
        read_from_variable="position_cleaned_lowconf",
        save_to_variable="position_egocentric_aligned",
    )
    outlier_cleaning(stepwise, read_from_variable="position_egocentric_aligned", save_to_variable="position_processed")
    savgol_filtering(stepwise, read_from_variable="position_processed", save_to_variable="position_processed")
    rescaling(stepwise, read_from_variable="position_processed", save_to_variable="position_scaled")

    vame.preprocessing(config=combined, centered_reference_keypoint=CENTER, orientation_reference_keypoint=ORIENTATION)

    assert_same_processed(stepwise, combined)

