import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr

from vame.io.load_poses import load_vame_dataset
from vame.logging.logger import VameLogger
from vame.util.auxiliary import write_config


logger_config = VameLogger(__name__)
logger = logger_config.logger


def add_extra_features(
    config: dict,
    session: str,
    features: Dict[str, np.ndarray],
    register_in_config: bool = True,
    config_path: Optional[str] = None,
) -> None:
    """
    Append pre-computed extra (non-pose) features to a session's processed .nc.

    Each value in ``features`` must be a 1-D numpy array whose length matches
    the session's ``position_processed`` time dimension. Features are written
    as data variables with dims ``("time", "individuals")`` and populated only
    at ``individuals[0]`` (multi-individual support deferred); other
    individuals are filled with NaN.

    The write is atomic: data is written to a temporary ``.nc`` file alongside
    the original, then renamed into place, so a failure mid-write cannot
    corrupt the existing file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    session : str
        Session name (must match an entry in ``config["session_names"]``).
    features : dict[str, np.ndarray]
        Mapping of feature name to a 1-D numeric array of length
        ``len(position_processed.time)``.
    register_in_config : bool, default True
        If True, append the feature names to ``config["extra_features"]``
        (de-duplicated, order-preserved) and persist the config to disk.
    config_path : str, optional
        Explicit path to ``config.yaml``. If omitted, defaults to
        ``{config["project_path"]}/config.yaml``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the session's processed ``.nc`` does not exist.
    ValueError
        If a feature array is not 1-D, has the wrong length, or is non-numeric.
    """
    if not features:
        return

    project_path = Path(config["project_path"])
    nc_path = project_path / "data" / "processed" / f"{session}_processed.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Processed file not found for session '{session}': {nc_path}")

    ds = load_vame_dataset(ds_path=nc_path)

    if "position_processed" not in ds.data_vars:
        raise ValueError(
            f"Session '{session}' has no 'position_processed' variable — run preprocessing first."
        )

    expected_time = ds.sizes["time"]
    n_individuals = ds.sizes.get("individuals", 1)

    for name, arr in features.items():
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(
                f"Feature '{name}' must be a 1-D array; got shape {arr.shape}. "
                "Multi-dim features should be split into separate scalar columns "
                "(e.g. 'head_dir_x', 'head_dir_y')."
            )
        if arr.shape[0] != expected_time:
            raise ValueError(
                f"Feature '{name}' has length {arr.shape[0]}; "
                f"expected {expected_time} (matching position_processed time)."
            )
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Feature '{name}' has non-numeric dtype {arr.dtype}.")

        data = np.full((expected_time, n_individuals), np.nan, dtype=np.float64)
        data[:, 0] = arr
        ds[name] = xr.DataArray(data, dims=("time", "individuals"))

    tmp_path = nc_path.with_suffix(".nc.tmp")
    ds.to_netcdf(path=tmp_path, engine="netcdf4")
    os.replace(tmp_path, nc_path)
    logger.info(
        f"Added {len(features)} extra feature(s) to {session}: {list(features.keys())}"
    )

    if register_in_config:
        existing = list(config.get("extra_features") or [])
        seen = set(existing)
        for name in features:
            if name not in seen:
                existing.append(name)
                seen.add(name)
        config["extra_features"] = existing

        cfg_path = config_path or str(project_path / "config.yaml")
        write_config(config_path=cfg_path, config=config)
