import numpy as np
import pytest
from pathlib import Path
from typing import Literal
from vame.model.create_training import traindata_aligned


def test_traindata_aligned_basic(setup_project_and_align_egocentric):
    """Test basic functionality of traindata_aligned with default parameters"""
    config = setup_project_and_align_egocentric["config_data"]

    # Call traindata_aligned with default parameters
    traindata_aligned(config=config)

    # Verify output files exist
    project_path = config["project_path"]
    train_path = Path(project_path) / "data" / "train" / "train_seq.npy"
    test_path = Path(project_path) / "data" / "train" / "test_seq.npy"

    assert train_path.exists()
    assert test_path.exists()

    # Load and verify data
    train_data = np.load(train_path)
    test_data = np.load(test_path)

    # Verify data shapes are consistent
    assert train_data.ndim == 2
    assert test_data.ndim == 2
    assert train_data.shape[0] == test_data.shape[0]  # Same number of features


@pytest.mark.parametrize("split_mode", ["mode_1", "mode_2"])
def test_traindata_aligned_split_modes(setup_project_and_align_egocentric, split_mode: Literal["mode_1", "mode_2"]):
    """Test both split modes produce valid outputs"""
    config = setup_project_and_align_egocentric["config_data"]
    test_fraction = 0.2  # Use 20% for testing

    # Call traindata_aligned with specific split mode
    traindata_aligned(
        config=config,
        test_fraction=test_fraction,
        split_mode=split_mode
    )

    # Load data
    project_path = config["project_path"]
    train_data = np.load(Path(project_path) / "data" / "train" / "train_seq.npy")
    test_data = np.load(Path(project_path) / "data" / "train" / "test_seq.npy")

    total_frames = train_data.shape[1] + test_data.shape[1]
    expected_test_frames = int(total_frames * test_fraction)

    # Allow for small rounding differences (±1 frame)
    assert abs(test_data.shape[1] - expected_test_frames) <= 1
    assert abs(train_data.shape[1] - (total_frames - expected_test_frames)) <= 1


def test_traindata_aligned_invalid_inputs(setup_project_and_align_egocentric):
    """Test handling of invalid inputs"""
    config = setup_project_and_align_egocentric["config_data"]

    # Test with empty sessions list
    with pytest.raises(ValueError, match="No sessions provided for training data creation"):
        traindata_aligned(config=config, sessions=[])

    # Test with invalid test_fraction values
    for invalid_fraction in [-0.1, 0, 1.0, 1.5]:
        traindata_aligned(config=config, test_fraction=invalid_fraction)
        # Should use default 0.1 test_fraction
        project_path = config["project_path"]
        test_data = np.load(Path(project_path) / "data" / "train" / "test_seq.npy")
        train_data = np.load(Path(project_path) / "data" / "train" / "train_seq.npy")
        total_frames = train_data.shape[1] + test_data.shape[1]
        assert abs(test_data.shape[1] - int(total_frames * 0.1)) <= 1


@pytest.mark.parametrize("test_fraction", [0.1, 0.2, 0.3])
def test_traindata_aligned_proportions(setup_project_and_align_egocentric, test_fraction):
    """Test if train/test split respects the test_fraction parameter"""
    config = setup_project_and_align_egocentric["config_data"]

    modes: list[Literal["mode_1", "mode_2"]] = ["mode_1", "mode_2"]
    for split_mode in modes:
        traindata_aligned(
            config=config,
            test_fraction=test_fraction,
            split_mode=split_mode
        )

        project_path = config["project_path"]
        train_data = np.load(Path(project_path) / "data" / "train" / "train_seq.npy")
        test_data = np.load(Path(project_path) / "data" / "train" / "test_seq.npy")

        total_frames = train_data.shape[1] + test_data.shape[1]
        expected_test_frames = int(total_frames * test_fraction)

        # Allow for small rounding differences (±1 frame)
        assert abs(test_data.shape[1] - expected_test_frames) <= 1
        assert abs(train_data.shape[1] - (total_frames - expected_test_frames)) <= 1


def test_traindata_aligned_data_continuity(setup_project_and_align_egocentric):
    """Test if data segments maintain temporal continuity in mode_2"""
    config = setup_project_and_align_egocentric["config_data"]
    test_fraction = 0.2

    traindata_aligned(
        config=config,
        test_fraction=test_fraction,
        split_mode="mode_2"  # type: Literal["mode_1", "mode_2"]
    )

    project_path = config["project_path"]
    train_data = np.load(Path(project_path) / "data" / "train" / "train_seq.npy")
    test_data = np.load(Path(project_path) / "data" / "train" / "test_seq.npy")

    # Verify data is not empty
    assert train_data.shape[1] > 0
    assert test_data.shape[1] > 0

    # Verify feature dimensions match
    assert train_data.shape[0] == test_data.shape[0]

    # For mode_2, each test chunk should be continuous
    # This means the data should not have sudden jumps
    # We can check this by computing the difference between consecutive frames
    # and verifying it's within a reasonable range
    test_diffs = np.abs(np.diff(test_data, axis=1))
    train_diffs = np.abs(np.diff(train_data, axis=1))

    # Get the median difference as a reference
    median_diff = np.median(test_diffs)

    # Check that most frame-to-frame differences are not much larger than the median
    # Allow for some outliers (using 95th percentile)
    assert np.percentile(test_diffs, 95) < median_diff * 10
    assert np.percentile(train_diffs, 95) < median_diff * 10
