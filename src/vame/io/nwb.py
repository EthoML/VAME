from typing import Optional
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import xarray as xr
import pynwb
import ndx_pose
import ndx_vame

from vame.logging.logger import VameLogger
from vame.io import load_vame_dataset
from vame.util.auxiliary import get_version


logger_config = VameLogger(__name__)
logger = logger_config.logger


def get_base_nwbfile(
    session_name: str,
    ds: xr.Dataset,
    nwbfile_kwargs: dict,
    subject_kwargs: dict,
):
    """
    Create base NWBFile object.
    """
    # Copy so the caller's dicts are left as they were
    subject_kwargs = dict(subject_kwargs)
    nwbfile_kwargs = dict(nwbfile_kwargs)
    subject_id = subject_kwargs.pop("subject_id", session_name)
    subject = pynwb.file.Subject(
        subject_id=subject_id,
        **subject_kwargs,
    )
    nwbfile = pynwb.NWBFile(
        session_description=nwbfile_kwargs.pop("session_description", "session_description"),
        identifier=nwbfile_kwargs.pop("identifier", session_name),
        session_start_time=nwbfile_kwargs.pop("session_start_time", datetime.now(timezone.utc)),
        subject=subject,
        **nwbfile_kwargs,
    )
    camera = nwbfile.create_device(
        name="camera",
        description="camera for recording behavior",
        manufacturer="my manufacturer",
    )

    # Create pose estimation and skeletons objects
    individuals = ds.individuals.values
    if len(individuals) > 1:
        raise ValueError("Multiple individuals are not supported yet.")
    keypoints = ds.keypoints.values
    pose_estimation_series_kwargs = {}
    if getattr(ds, "fps", None):
        pose_estimation_series_kwargs["rate"] = ds.fps
        pose_estimation_series_kwargs["starting_time"] = 0.
    else:
        pose_estimation_series_kwargs["timestamps"] = ds.sel(keypoints=keypoints[0]).time.values
    pose_estimation_series_kwargs["reference_frame"] = "(0,0,0) corresponds to ..."

    pose_estimation_series_list = []
    for keypoint in keypoints:
        pose_estimation_series_list.append(
            ndx_pose.PoseEstimationSeries(
                name=keypoint,
                data=ds.sel(keypoints=keypoint, individuals=individuals[0]).position.values,
                confidence=ds.sel(keypoints=keypoint, individuals=individuals[0]).confidence.values,
                unit="pixels",
                **pose_estimation_series_kwargs,
            )
        )

    skeleton = ndx_pose.Skeleton(
        name=f"{subject_id}_skeleton",
        nodes=keypoints,
        subject=subject,
    )
    skeletons = ndx_pose.Skeletons(skeletons=[skeleton])

    source_software = getattr(ds, "source_software", "Unknown Software")
    video_path = getattr(ds, "video_path", None)
    pose_estimation = ndx_pose.PoseEstimation(
        name="PoseEstimation",
        pose_estimation_series=pose_estimation_series_list,
        description=f"Estimated positions using {source_software}.",
        original_videos=[video_path],
        devices=[camera],
        source_software=source_software,
        skeleton=skeleton,
    )

    # Create behavior processing module
    behavior_pm = nwbfile.create_processing_module(
        name="behavior",
        description="processed behavioral data",
    )
    behavior_pm.add(skeletons)
    behavior_pm.add(pose_estimation)

    return nwbfile


def export_to_nwb(
    config: dict,
    nwbfile_kwargs: Optional[list[dict]] = None,
    subject_kwargs: Optional[list[dict]] = None,
):
    """
    Export VAME project to NWB format, one file per session at
    results/<session>/<model_name>/<session>.nwb, with the results of all segmentation algorithms.
    """
    session_names = config.get("session_names")
    if not session_names:
        raise ValueError("No session names provided in the config.")

    if nwbfile_kwargs is None:
        nwbfile_kwargs = [{}] * len(session_names)
    if len(nwbfile_kwargs) != len(session_names):
        raise ValueError("Number of nwbfile_kwargs must match number of sessions.")

    if subject_kwargs is None:
        subject_kwargs = [{}] * len(session_names)
    if len(subject_kwargs) != len(session_names):
        raise ValueError("Number of subject_kwargs must match number of sessions.")

    model_time_window = config["time_window"]
    vame_starting_sample_offset = int(model_time_window / 2)

    for session_name, sub, nwbmeta in zip(session_names, subject_kwargs, nwbfile_kwargs):
        # Load session data
        data_path = (Path(config["project_path"]) / "data" / "processed" / f"{session_name}_processed.nc").resolve()
        ds = load_vame_dataset(ds_path=data_path)
        if ds is None:
            raise ValueError(f"Dataset not found for session: {session_name}")

        if not getattr(ds, "fps", None):
            raise ValueError(
                f"Session {session_name} has no sampling rate, so its times are frame numbers. "
                "Exporting to NWB needs times in seconds: create the project with fps= or with videos."
            )
        vame_series_kwargs = {
            "rate": ds.fps,
            "starting_time": vame_starting_sample_offset / ds.fps,
        }

        # VAME content
        model_name = config.get("model_name")
        n_clusters = config.get("n_clusters")
        segmentation_algorithms = config.get("segmentation_algorithms", [])
        if not segmentation_algorithms:
            raise ValueError("No segmentation algorithms provided in the config.")

        # Latent space data
        latent_path = (
            Path(config["project_path"]) /
            "results" /
            session_name /
            model_name /
            "latent_vectors.npy"
        ).resolve()
        if not latent_path.exists():
            raise ValueError(f"Latent space data not found at: {latent_path}. Make sure to run vame.segment_session() first.")
        latent_data = np.load(latent_path)

        nwbfile = get_base_nwbfile(
            session_name=session_name,
            ds=ds,
            nwbfile_kwargs=nwbmeta,
            subject_kwargs=sub,
        )
        behavior_pm = nwbfile.processing["behavior"]
        latent_space_series = ndx_vame.LatentSpaceSeries(
            name="LatentSpaceSeries",
            description="VAME latent space vectors, one per sliding window.",
            data=latent_data,
            **vame_series_kwargs,
        )

        motif_series_list = []
        community_series_list = []
        for seg in segmentation_algorithms:
            seg_path = Path(config["project_path"]) / "results" / session_name / model_name / f"{seg}-{n_clusters}"
            motifs_path = (seg_path / f"{n_clusters}_{seg}_label_{session_name}.npy").resolve()
            if not motifs_path.exists():
                raise ValueError(f"Motif data not found at: {motifs_path}. Make sure to run vame.segment_session() first.")
            motif_series = ndx_vame.MotifSeries(
                name=f"MotifSeries{seg.capitalize()}",
                description=f"VAME motifs from {seg} with {n_clusters} clusters.",
                data=np.load(motifs_path),
                algorithm=seg,
                latent_space_series=latent_space_series,
                **vame_series_kwargs,
            )
            motif_series_list.append(motif_series)

            community_path = (seg_path / "community" / f"cohort_community_label_{session_name}.npy").resolve()
            if not community_path.exists():
                raise ValueError(f"Community data not found at: {community_path}. Make sure to run vame.community() first.")
            community_series_list.append(
                ndx_vame.CommunitySeries(
                    name=f"CommunitySeries{seg.capitalize()}",
                    description=f"VAME communities from the {seg} motifs.",
                    data=np.load(community_path),
                    motif_series=motif_series,
                    algorithm="hierarchical_clustering",
                    **vame_series_kwargs,
                )
            )

        vame_project = ndx_vame.VAMEProject(
            name="VAMEProject",
            pose_estimation=behavior_pm["PoseEstimation"],
            latent_space_series=latent_space_series,
            motif_series=motif_series_list,
            community_series=community_series_list,
            vame_config=json.dumps(config),
            time_window_samples=model_time_window,
            vame_version=get_version(),
        )
        behavior_pm.add(vame_project)

        nwbfile_path = (Path(config["project_path"]) / "results" / session_name / model_name / f"{session_name}.nwb").resolve()
        with pynwb.NWBHDF5IO(str(nwbfile_path), "w") as io:
            io.write(nwbfile)
        logger.info(f"{session_name} saved to NWB file at: {nwbfile_path}.")
