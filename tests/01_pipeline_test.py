import xarray as xr
from pathlib import Path

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

    # Test export to nwb
    export_to_nwb(config=pipeline.config)
    model_name = pipeline.config["model_name"]
    segmentation_algorithms = pipeline.config["segmentation_algorithms"]
    n_clusters = pipeline.config["n_clusters"]
    for session in sessions:
        nwbfile_path = Path(project_path) / "results" / session / model_name / f"{segmentation_algorithms[0]}-{n_clusters}" / f"{session}.nwb"
        assert nwbfile_path.exists()
