import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import vame
from vame.io.load_poses import load_vame_dataset
from vame.preprocessing import alignment, cleaning, scaling
from vame.preprocessing import filter as filtering
from vame.preprocessing.preprocessing import logger as preprocessing_logger
from vame.preprocessing.alignment import egocentrically_align_and_center
from vame.preprocessing.cleaning import lowconf_cleaning, outlier_cleaning
from vame.preprocessing.filter import savgol_filtering
from vame.preprocessing.scaling import rescaling, rescale_dataset
from vame.util.auxiliary import read_config

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


def test_rescaling_without_alignment_raises(tmp_path, pose_files):
    """Rescaling without alignment fails before preprocessing and leaves the config unchanged"""
    config = make_project(tmp_path, "no_alignment", pose_files)
    config_path = Path(config["project_path"]) / "config.yaml"
    config_before = read_config(str(config_path))

    with pytest.raises(ValueError, match="run_egocentric_alignment"):
        vame.preprocessing(
            config=config,
            centered_reference_keypoint=CENTER,
            orientation_reference_keypoint=ORIENTATION,
            run_egocentric_alignment=False,
            run_rescaling=True,
        )
    assert read_config(str(config_path)) == config_before


def test_rescale_dataset_without_individual_scale_raises():
    """A dataset without individual_scale raises instead of being skipped"""
    ds = xr.Dataset({"position": (("time", "space", "keypoints", "individuals"), np.ones((3, 2, 2, 1)))})
    with pytest.raises(KeyError, match="individual_scale"):
        rescale_dataset(ds=ds, read_from_variable="position", save_to_variable="position_scaled")
    assert "position_scaled" not in ds


def test_preprocessing_closes_log_files(tmp_path, pose_files):
    """Preprocessing removes the file handlers it adds to its loggers"""
    config = make_project(tmp_path, "logs", pose_files)
    vame.preprocessing(
        config=config,
        centered_reference_keypoint=CENTER,
        orientation_reference_keypoint=ORIENTATION,
        save_logs=True,
    )
    log_path = str(Path(config["project_path"]) / "logs" / "preprocessing.log")
    for logger in (preprocessing_logger, cleaning.logger, alignment.logger, filtering.logger, scaling.logger):
        assert not any(getattr(h, "baseFilename", None) == log_path for h in logger.handlers), logger.name
