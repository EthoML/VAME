from pathlib import Path
from typing import List, Optional

import numpy as np

from vame.io.load_poses import load_vame_dataset
from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger


def validate_extra_features(
    config: dict,
    sessions: Optional[List[str]] = None,
    extra_features: Optional[List[str]] = None,
) -> None:
    """
    Validate that every session's processed .nc satisfies the extra-feature contract.

    For each name in the extra-feature list and each session, checks:
      - The processed ``.nc`` file exists.
      - The data variable is present.
      - Its dims are ``(time,)`` or ``(time, individuals)``.
      - Its ``time`` length matches the session's ``position_processed`` time length.
      - Its dtype is numeric (float or int).
      - The first-individual slice contains no NaN or Inf values.

    All problems across all sessions are collected and reported in a single
    ``ValueError`` so users can fix them in one pass.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    sessions : list of str, optional
        Sessions to validate. Defaults to ``config["session_names"]``.
    extra_features : list of str, optional
        Override the list of feature names to validate. Defaults to
        ``config["extra_features"]``. Inference paths pass the list recovered
        from training metadata so they validate against what the model expects,
        not what the live config currently contains.

    Returns
    -------
    None
        Returns silently on success.

    Raises
    ------
    ValueError
        If any registered extra feature fails the contract for any session.
    """
    if extra_features is None:
        extra_features = list(config.get("extra_features") or [])
    else:
        extra_features = list(extra_features)
    if not extra_features:
        return

    if sessions is None:
        sessions = config["session_names"]

    project_path = Path(config["project_path"])
    problems: List[str] = []

    for session in sessions:
        nc_path = project_path / "data" / "processed" / f"{session}_processed.nc"
        if not nc_path.exists():
            problems.append(f"[{session}] processed file not found: {nc_path}")
            continue

        try:
            ds = load_vame_dataset(ds_path=nc_path)
        except Exception as e:
            problems.append(f"[{session}] failed to read {nc_path}: {e}")
            continue

        if "position_processed" not in ds.data_vars:
            problems.append(f"[{session}] missing 'position_processed' — run preprocessing first")
            continue
        expected_time = ds.sizes["time"]

        for name in extra_features:
            if name not in ds.data_vars:
                problems.append(f"[{session}] missing extra feature '{name}'")
                continue

            da = ds[name]

            if da.dims not in (("time",), ("time", "individuals")):
                problems.append(
                    f"[{session}] '{name}' has dims {da.dims}; expected ('time',) or ('time', 'individuals')"
                )
                continue

            if da.sizes["time"] != expected_time:
                problems.append(
                    f"[{session}] '{name}' has time length {da.sizes['time']}; "
                    f"expected {expected_time} (position_processed)"
                )
                continue

            if not np.issubdtype(da.dtype, np.number):
                problems.append(f"[{session}] '{name}' has non-numeric dtype {da.dtype}")
                continue

            values = da.isel(individuals=0).values if "individuals" in da.dims else da.values
            if not np.isfinite(values).all():
                problems.append(
                    f"[{session}] '{name}' contains NaN or Inf — VAME does not interpolate "
                    f"extra features; clean them before training"
                )

    if problems:
        msg = (
            f"validate_extra_features found {len(problems)} problem(s) across "
            f"{len(sessions)} session(s):\n  - " + "\n  - ".join(problems)
        )
        raise ValueError(msg)

    logger.info(
        f"validate_extra_features: {len(extra_features)} feature(s) verified across "
        f"{len(sessions)} session(s)."
    )
