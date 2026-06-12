import pytest

from vame.io.dandi import (
    is_remote_url,
    open_remote_nwbfile,
    _resolve_asset_urls,
    _find_pose_estimations,
    dandiset_load,
)
from vame.io.load_poses import load_vame_dataset

DANDISET = "000689"
VERSION = "0.240530.1923"
KEYPOINTS = ["HeadPoseEstimationSeries", "TailPoseEstimationSeries"]


def test_is_remote_url():
    assert is_remote_url("https://example.org/x.nwb")
    assert is_remote_url("s3://bucket/key")
    assert not is_remote_url("/local/path.nwb")
    assert not is_remote_url(None)


@pytest.fixture(scope="module")
def asset_urls():
    try:
        return _resolve_asset_urls(DANDISET, VERSION)
    except Exception as e:  # network/DANDI unavailable
        pytest.skip(f"DANDI not reachable: {e}")


@pytest.fixture(scope="module")
def two_files(asset_urls):
    """Two known sessions with ndx-pose data, taken from the resolved asset list."""
    picks = []
    for sub in ("sub-H12_ses-12", "sub-H29"):
        picks += [p for p in asset_urls if sub in p][:1]
    assert len(picks) == 2, f"expected 2 matching files, got {picks}"
    return picks


def test_resolve_asset_urls(asset_urls):
    assert asset_urls
    assert all(path.endswith(".nwb") for path in asset_urls)
    assert all(url.startswith("http") for url in asset_urls.values())


def test_find_pose_estimations(asset_urls, two_files):
    nwbfile, io = open_remote_nwbfile(asset_urls[two_files[0]])
    try:
        found = _find_pose_estimations(nwbfile)
    finally:
        io.close()
    assert found  # at least one PoseEstimation container
    series = {s for _, _, names in found for s in names}
    assert set(KEYPOINTS).issubset(series)


def test_dandiset_load(two_files, tmp_path):
    nc_paths = dandiset_load(
        DANDISET, VERSION,
        files=two_files,
        pose_estimation_series=KEYPOINTS,
        out_dir=str(tmp_path),
    )
    assert len(nc_paths) == len(two_files)
    for p in nc_paths:
        ds = load_vame_dataset(p)
        assert set(ds.keypoints.values.tolist()) == set(KEYPOINTS)
        assert "position" in ds and "confidence" in ds
        # Individual is named after the NWB subject_id (e.g. "H12")
        assert str(ds.individuals.values[0]).startswith("H")
