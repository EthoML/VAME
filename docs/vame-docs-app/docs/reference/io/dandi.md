---
sidebar_label: dandi
title: io.dandi
---

#### logger\_config

#### logger

#### REMOTE\_PREFIXES

#### is\_remote\_url

```python
def is_remote_url(path) -> bool
```

True if ``path`` is a remote URL string that should be streamed.

#### open\_remote\_nwbfile

```python
def open_remote_nwbfile(url: str) -> Tuple[pynwb.NWBFile, pynwb.NWBHDF5IO]
```

Open a remote NWB file for lazy streaming.

Returns ``(nwbfile, io)``. Keep ``io`` open until the data you need has been
read into memory, then close it.

#### \_short\_err

```python
def _short_err(exc: BaseException, limit: int = 400) -> str
```

Concise one-line error string. pynwb/hdmf errors (e.g. ConstructError) embed
the whole file&#x27;s builder tree, so we keep the type plus a bounded message
(head + tail, since the real cause is usually at the end).

#### \_resolve\_asset\_urls

```python
def _resolve_asset_urls(dandiset: str,
                        version: str,
                        files: Optional[List[str]] = None,
                        limit: Optional[int] = None) -> Dict[str, str]
```

Map asset path -&gt; stream URL for the ``.nwb`` assets of a dandiset.

If ``files`` is given, only those asset paths are resolved. If ``limit`` is
given, stop after collecting that many ``.nwb`` assets.

#### \_find\_pose\_estimations

```python
def _find_pose_estimations(
        nwbfile: pynwb.NWBFile) -> List[Tuple[str, str, List[str]]]
```

Locate ndx-pose PoseEstimation containers in the file.

Scans all processing modules and returns one entry per
``ndx_pose.PoseEstimation`` as ``(processing_module_key, pose_estimation_key,
[series_names])``.

#### \_load\_movement\_ds

```python
def _load_movement_ds(nwbfile: pynwb.NWBFile,
                      requested_series: List[str],
                      progress_prefix: str = "") -> xr.Dataset
```

Build a movement dataset from the file, keeping only the requested series.

Dataset construction is delegated to movement&#x27;s ``from_nwb_file`` (one call
per PoseEstimation container, merged); series selection is applied afterward
as an xarray subset. ``from_nwb_file`` reads all keypoints, so selection here
is a post-filter, not a fetch-filter. The NWB file must still be open.

#### dandiset\_parse

```python
def dandiset_parse(dandiset: str,
                   version: str,
                   n_items: Optional[int] = None) -> dict
```

Scan a dandiset for NWB files containing ndx-pose PoseEstimation data.

Each ``.nwb`` asset is streamed open (metadata only) and inspected for
``ndx_pose.PoseEstimation`` containers across all processing modules.

**Parameters**

* **dandiset** (`str`): Dandiset identifier, e.g. ``&quot;000689&quot;``.
* **version** (`str`): Published version (e.g. ``&quot;0.240530.1923&quot;``) or ``&quot;draft&quot;``.
* **n_items** (`int, optional`): If given, scan only the first ``n_items`` NWB assets (handy for quick
checks/tests). Default ``None`` scans the entire dandiset.

**Returns**

* `dict`: ``{&quot;valid_files&quot;: [...], &quot;pose_estimation_series&quot;: [{&quot;name&quot;, &quot;count&quot;}, ...]}``
where ``valid_files`` are asset paths that contain PoseEstimation data,
and each entry of ``pose_estimation_series`` is a series (keypoint) name
with the number of valid files it appears in, ordered by count desc.

#### dandiset\_load

```python
def dandiset_load(dandiset: str,
                  version: str,
                  files: List[str],
                  pose_estimation_series: List[str],
                  out_dir: Path | str = ".") -> List[str]
```

Stream selected files/series from a dandiset into local movement ``.nc`` files.

Only the requested files and the requested pose series (keypoints) are
fetched. The resulting ``.nc`` files can be passed straight to
``vame.init_new_project(..., source_software=&quot;movement&quot;)``.

**Parameters**

* **dandiset** (`str`): Dandiset identifier, e.g. ``&quot;000689&quot;``.
* **version** (`str`): Published version or ``&quot;draft&quot;``.
* **files** (`List[str]`): Asset paths to load (e.g. the ``valid_files`` from :func:`dandiset_parse`).
* **pose_estimation_series** (`List[str]`): Series (keypoint) names to include; others are not fetched.
* **out_dir** (`Path or str, optional`): Directory to write the ``.nc`` files into (created if missing). Default &quot;.&quot;.

**Returns**

* `List[str]`: Paths to the written ``.nc`` files, in the order of ``files``.

