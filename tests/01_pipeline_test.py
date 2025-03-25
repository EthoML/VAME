import xarray as xr
from pathlib import Path


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
    assert save_fig_path_0.exists()
    assert save_fig_path_1.exists()
