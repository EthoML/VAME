from pathlib import Path

import pytest

from vame.io.dandi import (
    is_remote_url,
    open_remote_nwbfile,
    _resolve_asset_urls,
    _find_pose_estimations,
    dandiset_parse,
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


class FakeAsset:
    def __init__(self, path):
        self.path = path

    def get_content_url(self, follow_redirects, strip_query):
        url = f"https://bucket.s3.amazonaws.com/{self.path}"
        return url if strip_query else f"{url}?X-Amz-Signature=signed"


class FakeClient:
    def __init__(self, instance_name):
        self.dandi_instance = type("Instance", (), {"name": instance_name})()
        self.api_key_env_var = f"{instance_name.upper().replace('-', '_')}_API_KEY"
        self.session = type("Session", (), {"headers": {}})()

    def dandi_authenticate(self):
        self.session.headers["Authorization"] = "token from-env"

    def get_dandiset(self, dandiset, version):
        assets = [FakeAsset("sub-1/sub-1_ses-1.nwb"), FakeAsset("sub-1/sub-1_ses-1_image/video.avi")]
        return type("Dandiset", (), {"get_assets": lambda self: iter(assets)})()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def fake_clients(monkeypatch):
    from dandi.dandiapi import DandiAPIClient

    created = []

    def for_dandi_instance(instance):
        created.append(FakeClient(instance))
        return created[-1]

    monkeypatch.setattr(DandiAPIClient, "for_dandi_instance", staticmethod(for_dandi_instance))
    monkeypatch.delenv("DANDI_API_KEY", raising=False)
    monkeypatch.delenv("DANDI_SANDBOX_API_KEY", raising=False)
    return created


def test_resolve_asset_urls_unauthenticated_strips_query(fake_clients):
    urls = _resolve_asset_urls("123456", "draft")
    assert fake_clients[0].dandi_instance.name == "dandi"
    assert urls == {"sub-1/sub-1_ses-1.nwb": "https://bucket.s3.amazonaws.com/sub-1/sub-1_ses-1.nwb"}


def test_resolve_asset_urls_authenticated_sandbox_keeps_signed_url(fake_clients, monkeypatch):
    monkeypatch.setenv("DANDI_SANDBOX_API_KEY", "secret")
    urls = _resolve_asset_urls("123456", "draft", dandi_instance="dandi-sandbox")
    assert fake_clients[0].dandi_instance.name == "dandi-sandbox"
    assert "Authorization" in fake_clients[0].session.headers
    assert urls["sub-1/sub-1_ses-1.nwb"].endswith("?X-Amz-Signature=signed")


def test_resolve_asset_urls_unknown_instance(fake_clients):
    with pytest.raises(ValueError, match="Unknown DANDI instance 'dandi-sanbox'.*dandi-sandbox"):
        _resolve_asset_urls("123456", "draft", dandi_instance="dandi-sanbox")
    assert fake_clients == []


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


def test_dandiset_parse_n_items():
    # `limit` stops the resolver early — exactly n assets.
    assert len(_resolve_asset_urls(DANDISET, VERSION, limit=2)) == 2

    # `n_items` scans only the first few files and returns a consistent summary.
    result = dandiset_parse(DANDISET, VERSION, n_items=3)
    assert set(result) == {"valid_files", "pose_estimation_series"}
    assert len(result["valid_files"]) <= 3
    assert all(p.endswith(".nwb") for p in result["valid_files"])
    for s in result["pose_estimation_series"]:
        assert set(s) == {"name", "count"}
        assert 1 <= s["count"] <= len(result["valid_files"])


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


def test_dandiset_load_download(two_files, tmp_path):
    streamed = dandiset_load(
        DANDISET, VERSION,
        files=two_files[:1],
        pose_estimation_series=KEYPOINTS,
        out_dir=str(tmp_path / "streamed"),
    )
    downloaded = dandiset_load(
        DANDISET, VERSION,
        files=two_files[:1],
        pose_estimation_series=KEYPOINTS,
        out_dir=str(tmp_path / "downloaded"),
        download=True,
    )
    assert [Path(p).name for p in downloaded] == [Path(p).name for p in streamed]
    assert [p.suffix for p in (tmp_path / "downloaded").iterdir()] == [".nc"]
    assert load_vame_dataset(downloaded[0]).equals(load_vame_dataset(streamed[0]))
