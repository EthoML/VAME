import os
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter
import xarray as xr
import pynwb
import ndx_pose
from movement.io.load_poses import from_nwb_file

from vame.io.load_poses import _normalize_movement_dims
from vame.logging.logger import VameLogger

logger_config = VameLogger(__name__)
logger = logger_config.logger

REMOTE_PREFIXES = ("http://", "https://", "s3://", "dandi://")


def is_remote_url(path) -> bool:
    """True if ``path`` is a remote URL string that should be streamed."""
    return isinstance(path, str) and path.startswith(REMOTE_PREFIXES)


def open_remote_nwbfile(url: str) -> Tuple[pynwb.NWBFile, pynwb.NWBHDF5IO]:
    """
    Open a remote NWB file for lazy streaming.

    Returns ``(nwbfile, io)``. Keep ``io`` open until the data you need has been
    read into memory, then close it.
    """
    try:
        import remfile
    except ImportError as e:
        raise ImportError(
            "Streaming remote NWB files requires the 'remfile' package. "
            "Install it with: pip install remfile (or pip install vame-py[dandi])."
        ) from e
    import h5py

    h5 = h5py.File(remfile.File(url), "r")
    io = pynwb.NWBHDF5IO(file=h5, mode="r")
    return io.read(), io


def _short_err(exc: BaseException, limit: int = 400) -> str:
    """
    Concise one-line error string. pynwb/hdmf errors (e.g. ConstructError) embed
    the whole file's builder tree, so we keep the type plus a bounded message
    (head + tail, since the real cause is usually at the end).
    """
    msg = str(exc)
    if len(msg) > limit:
        head, tail = 80, limit - 80
        msg = f"{msg[:head]} …[+{len(msg) - limit} chars]… {msg[-tail:]}"
    return f"{type(exc).__name__}: {msg}"


def _get_client(dandi_instance: str = "dandi"):
    """
    Client for a DANDI instance (e.g. ``"dandi"`` or ``"dandi-sandbox"``).

    It authenticates when the instance's API key environment variable is set
    (``DANDI_API_KEY`` for ``"dandi"``, ``DANDI_SANDBOX_API_KEY`` for ``"dandi-sandbox"``),
    which is required to read embargoed dandisets.
    """
    try:
        from dandi.consts import known_instances
        from dandi.dandiapi import DandiAPIClient
    except ImportError as e:
        raise ImportError(
            "Reading from DANDI requires the 'dandi' package. "
            "Install it with: pip install dandi (or pip install vame-py[dandi])."
        ) from e

    if dandi_instance not in known_instances:
        raise ValueError(f"Unknown DANDI instance '{dandi_instance}'. Known instances: {list(known_instances)}")
    client = DandiAPIClient.for_dandi_instance(dandi_instance)
    if os.environ.get(client.api_key_env_var):
        client.dandi_authenticate()
        logger.info(f"Authenticated to {dandi_instance} with {client.api_key_env_var}.")
    return client


def _resolve_assets(
    client,
    dandiset: str,
    version: str,
    files: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict:
    """
    Map asset path -> asset for the ``.nwb`` assets of a dandiset.

    If ``files`` is given, only those asset paths are resolved. If ``limit`` is
    given, stop after collecting that many ``.nwb`` assets.
    """
    wanted = set(files) if files is not None else None
    assets = {}
    logger.info(f"Resolving .nwb assets from dandiset {dandiset} ({version}) on {client.dandi_instance.name}...")
    for asset in client.get_dandiset(dandiset, version).get_assets():
        if not asset.path.endswith(".nwb"):
            continue
        if wanted is not None and asset.path not in wanted:
            continue
        assets[asset.path] = asset
        if limit is not None and len(assets) >= limit:
            break
    logger.info(f"Resolved {len(assets)} .nwb asset(s).")
    return assets


def _stream_url(client, asset) -> str:
    """S3 URL to stream an asset from."""
    # Embargoed assets are only reachable through a signed URL, so the query is kept when authenticated
    authenticated = "Authorization" in client.session.headers
    return asset.get_content_url(follow_redirects=1, strip_query=not authenticated)


def _resolve_asset_urls(
    dandiset: str,
    version: str,
    files: Optional[List[str]] = None,
    limit: Optional[int] = None,
    dandi_instance: str = "dandi",
) -> Dict[str, str]:
    """
    Map asset path -> stream URL for the ``.nwb`` assets of a dandiset.

    If ``files`` is given, only those asset paths are resolved. If ``limit`` is
    given, stop after collecting that many ``.nwb`` assets.
    """
    with _get_client(dandi_instance) as client:
        assets = _resolve_assets(client, dandiset, version, files=files, limit=limit)
        return {path: _stream_url(client, asset) for path, asset in assets.items()}


def _open_asset(
    client,
    asset,
    download_dir: Optional[Path] = None,
) -> Tuple[pynwb.NWBFile, pynwb.NWBHDF5IO]:
    """
    Open an NWB asset by streaming it, or by downloading it into ``download_dir`` when given.

    Returns ``(nwbfile, io)``; close ``io`` when done.
    """
    if download_dir is None:
        return open_remote_nwbfile(_stream_url(client, asset))
    local_path = Path(download_dir) / Path(asset.path).name
    asset.download(local_path)
    io = pynwb.NWBHDF5IO(local_path, mode="r")
    return io.read(), io


def _find_pose_estimations(
    nwbfile: pynwb.NWBFile,
) -> List[Tuple[str, str, List[str]]]:
    """
    Locate ndx-pose PoseEstimation containers in the file.

    Scans all processing modules and returns one entry per
    ``ndx_pose.PoseEstimation`` as ``(processing_module_key, pose_estimation_key,
    [series_names])``.
    """
    found: List[Tuple[str, str, List[str]]] = []
    for pm_key, pm in nwbfile.processing.items():
        for obj_key, obj in pm.data_interfaces.items():
            if isinstance(obj, ndx_pose.PoseEstimation):
                found.append((pm_key, obj_key, list(obj.pose_estimation_series.keys())))
    return found


def _load_movement_ds(
    nwbfile: pynwb.NWBFile,
    requested_series: List[str],
    progress_prefix: str = "",
) -> xr.Dataset:
    """
    Build a movement dataset from the file, keeping only the requested series.

    Dataset construction is delegated to movement's ``from_nwb_file`` (one call
    per PoseEstimation container, merged); series selection is applied afterward
    as an xarray subset. ``from_nwb_file`` reads all keypoints, so selection here
    is a post-filter, not a fetch-filter. The NWB file must still be open.
    """
    pes_list = _find_pose_estimations(nwbfile)
    if not pes_list:
        raise ValueError("No ndx-pose PoseEstimation data found in the file.")

    datasets = []
    for pm_key, pe_key, names in pes_list:
        logger.info(f"{progress_prefix}reading {len(names)} keypoint(s) from {pm_key}/{pe_key} ...")
        datasets.append(from_nwb_file(nwbfile, processing_module_key=pm_key, pose_estimation_key=pe_key))
    ds = datasets[0] if len(datasets) == 1 else xr.merge(datasets, join="outer", compat="no_conflicts")
    ds = _normalize_movement_dims(ds)  # movement 0.17 uses singular dim names

    available = ds.coords["keypoints"].values.tolist()
    keep = [s for s in requested_series if s in available]
    if not keep:
        raise ValueError("None of the requested pose_estimation_series were found in the file.")
    ds = ds.sel(keypoints=keep)

    # Name the individual by NWB subject_id (the animal), not movement's default
    # (the file UUID). Fall back to the VAME/DLC convention when subject is absent.
    if ds.sizes.get("individuals", 0) == 1:
        subject = getattr(nwbfile, "subject", None)
        subject_id = getattr(subject, "subject_id", None) if subject is not None else None
        ds = ds.assign_coords(individuals=[subject_id or "individual_0"])

    return ds


def dandiset_parse(
    dandiset: str,
    version: str,
    n_items: Optional[int] = None,
    dandi_instance: str = "dandi",
) -> dict:
    """
    Scan a dandiset for NWB files containing ndx-pose PoseEstimation data.

    Each ``.nwb`` asset is streamed open (metadata only) and inspected for
    ``ndx_pose.PoseEstimation`` containers across all processing modules.

    Parameters
    ----------
    dandiset : str
        Dandiset identifier, e.g. ``"000689"``.
    version : str
        Published version (e.g. ``"0.240530.1923"``) or ``"draft"``.
    n_items : int, optional
        If given, scan only the first ``n_items`` NWB assets (handy for quick
        checks/tests). Default ``None`` scans the entire dandiset.
    dandi_instance : str, optional
        DANDI instance hosting the dandiset, e.g. ``"dandi"`` (default) or ``"dandi-sandbox"``.
        Embargoed dandisets need the instance's API key in the environment
        (``DANDI_API_KEY`` for ``"dandi"``, ``DANDI_SANDBOX_API_KEY`` for ``"dandi-sandbox"``).

    Returns
    -------
    dict
        ``{"valid_files": [...], "pose_estimation_series": [{"name", "count"}, ...]}``
        where ``valid_files`` are asset paths that contain PoseEstimation data,
        and each entry of ``pose_estimation_series`` is a series (keypoint) name
        with the number of valid files it appears in, ordered by count desc.
    """
    urls = _resolve_asset_urls(dandiset, version, limit=n_items, dandi_instance=dandi_instance)
    total = len(urls)
    valid_files: List[str] = []
    counter: Counter = Counter()
    skipped = 0
    logger.info(f"Scanning {total} .nwb file(s) for PoseEstimation data...")

    for i, (path, url) in enumerate(urls.items(), start=1):
        logger.info(f"[{i}/{total}] Opening {path} ...")
        try:
            nwbfile, io = open_remote_nwbfile(url)
        except Exception as e:
            skipped += 1
            logger.warning(f"[{i}/{total}] Skipping {path}: failed to open ({_short_err(e)})")
            continue
        try:
            pes_list = _find_pose_estimations(nwbfile)
        finally:
            io.close()
        series_names = {s for _, _, names in pes_list for s in names}
        if series_names:
            valid_files.append(path)
            counter.update(series_names)
            logger.info(f"[{i}/{total}] {path}: {len(series_names)} pose series")
        else:
            logger.info(f"[{i}/{total}] {path}: no PoseEstimation data")

    logger.info(
        f"Scan complete: {len(valid_files)}/{total} file(s) contain PoseEstimation data; "
        f"{len(counter)} unique series; {skipped} file(s) skipped (errors)."
    )
    return {
        "valid_files": valid_files,
        "pose_estimation_series": [{"name": n, "count": c} for n, c in counter.most_common()],
    }


def dandiset_load(
    dandiset: str,
    version: str,
    files: List[str],
    pose_estimation_series: List[str],
    out_dir: Path | str = ".",
    dandi_instance: str = "dandi",
    download: bool = False,
) -> List[str]:
    """
    Stream selected files/series from a dandiset into local movement ``.nc`` files.

    Only the requested files and the requested pose series (keypoints) are
    kept. The resulting ``.nc`` files can be passed straight to
    ``vame.init_new_project(..., source_software="movement")``.

    Parameters
    ----------
    dandiset : str
        Dandiset identifier, e.g. ``"000689"``.
    version : str
        Published version or ``"draft"``.
    files : List[str]
        Asset paths to load (e.g. the ``valid_files`` from :func:`dandiset_parse`).
    pose_estimation_series : List[str]
        Series (keypoint) names to include; others are not fetched.
    out_dir : Path or str, optional
        Directory to write the ``.nc`` files into (created if missing). Default ".".
    dandi_instance : str, optional
        DANDI instance hosting the dandiset, e.g. ``"dandi"`` (default) or ``"dandi-sandbox"``.
        Embargoed dandisets need the instance's API key in the environment
        (``DANDI_API_KEY`` for ``"dandi"``, ``DANDI_SANDBOX_API_KEY`` for ``"dandi-sandbox"``).
    download : bool, optional
        If True, download each NWB file to a temporary folder and read it locally, deleting it
        afterwards. Faster than streaming for files that hold little besides pose data.
        Default False streams the files.

    Returns
    -------
    List[str]
        Paths to the written ``.nc`` files, in the order of ``files``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with _get_client(dandi_instance) as client, tempfile.TemporaryDirectory() as tmp_dir:
        download_dir = tmp_dir if download else None
        assets = _resolve_assets(client, dandiset, version, files=files)
        missing = [f for f in files if f not in assets]
        if missing:
            logger.warning(f"{len(missing)} requested file(s) not found in dandiset and skipped: {missing}")

        to_load = [f for f in files if f in assets]
        total = len(to_load)
        logger.info(f"Loading {total} file(s) into movement .nc at {out} ...")

        nc_paths: List[str] = []
        for i, path in enumerate(to_load, start=1):
            logger.info(f"[{i}/{total}] {'Downloading' if download else 'Streaming'} {path} ...")
            try:
                nwbfile, io = _open_asset(client, assets[path], download_dir=download_dir)
            except Exception as e:
                logger.warning(f"[{i}/{total}] Skipping {path}: failed to open ({_short_err(e)})")
                continue
            try:
                ds = _load_movement_ds(nwbfile, pose_estimation_series, progress_prefix=f"[{i}/{total}] ")
            except ValueError as e:
                logger.warning(f"[{i}/{total}] {path}: {e}; skipping")
                continue
            finally:
                io.close()
                (Path(tmp_dir) / Path(path).name).unlink(missing_ok=True)

            name = path.replace("/", "_").removesuffix(".nwb")
            nc_path = out / f"{name}.nc"
            ds.to_netcdf(nc_path, engine="netcdf4")
            nc_paths.append(str(nc_path))
            logger.info(f"[{i}/{total}] Saved {path} ({ds.sizes['keypoints']} series) -> {nc_path}")

    logger.info(f"Load complete: wrote {len(nc_paths)}/{total} .nc file(s) to {out}.")
    return nc_paths
