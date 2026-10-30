from pathlib import Path
import numpy as np
import pynwb
import pytest

from vame.io import export_to_nwb
from vame.io.load_poses import load_pose_estimation

SAMPLE_CSV = Path(__file__).parent / "tests_project_sample_data" / "cropped_video.csv"
SESSION = "cropped_video"
TIME_WINDOW = 30
N_CLUSTERS = 3
ALGORITHMS = ["hmm", "kmeans"]


def _make_project(tmp_path, fps):
    """Minimal project with fake segmentation results, enough for export_to_nwb"""
    ds = load_pose_estimation(pose_estimation_file=str(SAMPLE_CSV), source_software="DeepLabCut", fps=fps)
    if fps is None:
        ds.attrs.pop("fps", None)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    ds.to_netcdf(tmp_path / "data" / "processed" / f"{SESSION}_processed.nc", engine="netcdf4")

    n_windows = ds.sizes["time"] - TIME_WINDOW + 1
    model_path = tmp_path / "results" / SESSION / "VAME"
    model_path.mkdir(parents=True)
    np.save(model_path / "latent_vectors.npy", np.zeros((n_windows, 4), dtype="float32"))
    labels = {}
    for i, seg in enumerate(ALGORITHMS):
        seg_path = model_path / f"{seg}-{N_CLUSTERS}"
        (seg_path / "community").mkdir(parents=True)
        labels[seg] = (np.arange(n_windows) + i) % N_CLUSTERS
        np.save(seg_path / f"{N_CLUSTERS}_{seg}_label_{SESSION}.npy", labels[seg])
        np.save(seg_path / "community" / f"cohort_community_label_{SESSION}.npy", labels[seg] % 2)
    config = {
        "project_path": str(tmp_path),
        "session_names": [SESSION],
        "time_window": TIME_WINDOW,
        "model_name": "VAME",
        "n_clusters": N_CLUSTERS,
        "segmentation_algorithms": ALGORITHMS,
    }
    return config, model_path / f"{SESSION}.nwb", labels


def test_export_to_nwb_one_file_with_all_algorithms(tmp_path):
    config, nwb_path, labels = _make_project(tmp_path, fps=25.0)
    subject_kwargs = [{"subject_id": "m1", "species": "Mus musculus"}]
    nwbfile_kwargs = [{"session_description": "open field", "experimenter": ["Doe, Jane"]}]
    export_to_nwb(config=config, nwbfile_kwargs=nwbfile_kwargs, subject_kwargs=subject_kwargs)

    assert subject_kwargs == [{"subject_id": "m1", "species": "Mus musculus"}]
    assert nwbfile_kwargs == [{"session_description": "open field", "experimenter": ["Doe, Jane"]}]
    assert list(Path(config["project_path"]).rglob("*.nwb")) == [nwb_path]
    with pynwb.NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()
        assert nwbfile.subject.subject_id == "m1"
        assert nwbfile.session_description == "open field"
        assert tuple(nwbfile.experimenter) == ("Doe, Jane",)
        vame_project = nwbfile.processing["behavior"]["VAMEProject"]
        assert vame_project.time_window_samples == TIME_WINDOW
        for seg in ALGORITHMS:
            motif_series = vame_project.motif_series[f"MotifSeries{seg.capitalize()}"]
            community_series = vame_project.community_series[f"CommunitySeries{seg.capitalize()}"]
            assert motif_series.algorithm == seg
            np.testing.assert_array_equal(motif_series.data[:], labels[seg])
            assert community_series.motif_series is motif_series
            assert motif_series.latent_space_series is vame_project.latent_space_series


def test_export_to_nwb_without_fps_raises(tmp_path):
    """A project without fps has times in frames, which NWB cannot store as seconds"""
    config, nwb_path, _ = _make_project(tmp_path, fps=None)
    with pytest.raises(ValueError, match="no sampling rate"):
        export_to_nwb(config=config)
    assert not nwb_path.exists()
