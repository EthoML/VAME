import numpy as np
import xarray as xr
from pathlib import Path

from vame.pipeline import VAMEPipeline
from vame.io import export_to_nwb


def test_pipeline(setup_pipeline):
    pipeline = setup_pipeline["pipeline"]
    project_path = pipeline.config["project_path"]
    sessions = pipeline.get_sessions()
    assert len(sessions) == 1

    ds = pipeline.get_raw_datasets()
    assert isinstance(ds, xr.Dataset)

    preprocessing_kwargs = {
        "centered_reference_keypoint": "Nose",
        "orientation_reference_keypoint": "Tailroot",
        "run_rescaling": True,
    }
    pipeline.run_pipeline(preprocessing_kwargs=preprocessing_kwargs)

    pipeline.visualize_preprocessing(
        show_figure=False,
        save_to_file=True,
    )
    save_fig_path_0 = Path(project_path) / "reports" / "figures" / f"{sessions[0]}_preprocessing_scatter.png"
    save_fig_path_1 = Path(project_path) / "reports" / "figures" / f"{sessions[0]}_preprocessing_timeseries.png"
    save_fig_path_2 = Path(project_path) / "reports" / "figures" / f"{sessions[0]}_preprocessing_cloud.png"
    assert save_fig_path_0.exists()
    assert save_fig_path_1.exists()
    assert save_fig_path_2.exists()

    # Visualizations
    pipeline.visualize_motif_thresholding(
        show_figure=False,
        save_to_file=True,
    )
    save_fig_path_3 = Path(project_path) / "reports" / "figures" / f"motif_thresholding_hmm_{pipeline.config['n_clusters']}.png"
    assert save_fig_path_3.exists()

    # Test export to nwb
    export_to_nwb(config=pipeline.config)
    model_name = pipeline.config["model_name"]
    for session in sessions:
        nwbfile_path = Path(project_path) / "results" / session / model_name / f"{session}.nwb"
        assert nwbfile_path.exists()


def test_visualize_model_losses_uses_config_model_name(tmp_path):
    """The loss plot reads the losses of the model named in the config"""
    losses_dir = tmp_path / "model" / "model_losses"
    losses_dir.mkdir(parents=True)
    (tmp_path / "model" / "evaluate").mkdir()
    for prefix in ("train", "test", "mse_train", "mse_test", "kmeans", "kl", "fut"):
        np.save(losses_dir / f"{prefix}_losses_custom.npy", np.ones(3))
    pipeline = VAMEPipeline.__new__(VAMEPipeline)
    pipeline.config = {"project_path": str(tmp_path), "model_name": "custom"}
    pipeline.visualize_model_losses(show_figure=False)
    assert (tmp_path / "model" / "evaluate" / "mse_and_kl_loss_custom.png").exists()
