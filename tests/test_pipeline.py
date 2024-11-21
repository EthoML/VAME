from pytest import fixture
from pathlib import Path
import shutil
import xarray as xr
from vame.pipeline import VAMEPipeline


@fixture(scope="session")
def setup_pipeline():
    """
    Setup a Pipeline for testing.
    """
    project_name = "test_pipeline"
    videos = ["./tests/tests_project_sample_data"]
    poses_estimations = ["./tests/tests_project_sample_data"]
    working_directory = "./tests"
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
    shutil.rmtree(Path(pipeline.config_path).parent)


def test_pipeline(setup_pipeline):
    pipeline = setup_pipeline["pipeline"]

    sessions = pipeline.get_sessions()
    assert len(sessions) == 1

    ds = pipeline.get_raw_datasets()
    assert isinstance(ds, xr.Dataset)

    pipeline.run_pipeline()
