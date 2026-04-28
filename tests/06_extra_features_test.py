from pathlib import Path

import numpy as np
import pytest

import vame
from vame.io.load_poses import load_vame_dataset
from vame.util.auxiliary import write_config


@pytest.fixture(scope="function")
def extras_project(setup_project_and_align_egocentric):
    """
    Function-scoped wrapper around the session-scoped preprocessed project.

    Snapshots ``config["extra_features"]`` (in-memory and on disk) at entry
    and restores both at teardown so that mutations from these tests don't
    pollute the session fixture when other test files run after this one.
    """
    project_data = setup_project_and_align_egocentric
    config = project_data["config_data"]
    config_path = project_data["config_path"]
    saved_extras = list(config.get("extra_features") or [])
    yield project_data
    config["extra_features"] = saved_extras
    write_config(config_path, config)


def _processed_nc_path(config: dict, session: str) -> Path:
    return Path(config["project_path"]) / "data" / "processed" / f"{session}_processed.nc"


def _session_time_length(config: dict, session: str) -> int:
    return load_vame_dataset(_processed_nc_path(config, session)).sizes["time"]


# --- writer -----------------------------------------------------------------


def test_add_extra_features_writes_var_and_registers(extras_project):
    config = extras_project["config_data"]
    session = config["session_names"][0]
    n_time = _session_time_length(config, session)
    speed = np.sin(np.linspace(0, 2 * np.pi, n_time)).astype(np.float64)

    vame.io.add_extra_features(
        config=config,
        session=session,
        features={"test_speed": speed},
    )

    ds = load_vame_dataset(_processed_nc_path(config, session))
    assert "test_speed" in ds.data_vars
    np.testing.assert_array_equal(ds["test_speed"].isel(individuals=0).values, speed)
    assert "test_speed" in config["extra_features"]


def test_add_extra_features_dedup(extras_project):
    config = extras_project["config_data"]
    session = config["session_names"][0]
    n_time = _session_time_length(config, session)
    values = np.zeros(n_time, dtype=np.float64)

    vame.io.add_extra_features(config=config, session=session, features={"test_dedup": values})
    vame.io.add_extra_features(config=config, session=session, features={"test_dedup": values})

    assert config["extra_features"].count("test_dedup") == 1


def test_add_extra_features_length_mismatch_raises(extras_project):
    config = extras_project["config_data"]
    session = config["session_names"][0]
    n_time = _session_time_length(config, session)
    bad = np.zeros(n_time + 1, dtype=np.float64)

    with pytest.raises(ValueError, match="length"):
        vame.io.add_extra_features(
            config=config,
            session=session,
            features={"test_length_bad": bad},
        )

    assert "test_length_bad" not in config.get("extra_features", [])


def test_add_extra_features_non_numeric_raises(extras_project):
    config = extras_project["config_data"]
    session = config["session_names"][0]
    n_time = _session_time_length(config, session)
    bad = np.array(["a"] * n_time, dtype=object)

    with pytest.raises(ValueError, match="non-numeric"):
        vame.io.add_extra_features(
            config=config,
            session=session,
            features={"test_non_numeric": bad},
        )


def test_add_extra_features_non_1d_raises(extras_project):
    config = extras_project["config_data"]
    session = config["session_names"][0]
    n_time = _session_time_length(config, session)
    bad = np.zeros((n_time, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="1-D"):
        vame.io.add_extra_features(
            config=config,
            session=session,
            features={"test_2d": bad},
        )


# --- validator --------------------------------------------------------------


def test_validate_extra_features_passes_after_write(extras_project):
    config = extras_project["config_data"]
    session = config["session_names"][0]
    n_time = _session_time_length(config, session)
    values = np.linspace(0, 1, n_time, dtype=np.float64)

    vame.io.add_extra_features(
        config=config,
        session=session,
        features={"test_validate": values},
    )

    vame.validate_extra_features(config)


def test_validate_extra_features_reports_missing_var(extras_project):
    config = extras_project["config_data"]
    session = config["session_names"][0]

    with pytest.raises(ValueError) as exc:
        vame.validate_extra_features(
            config=config,
            extra_features=["nonexistent_var"],
        )

    msg = str(exc.value)
    assert "nonexistent_var" in msg
    assert session in msg
