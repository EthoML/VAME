from typing import Optional
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import xarray as xr
import pynwb
import ndx_pose
import ndx_vame

from vame.io import load_vame_dataset


def get_base_nwbfile(
    session_name: str,
    ds: xr.Dataset,
    nwbfile_kwargs: dict,
    subject_kwargs: dict,
):
    """
    Create base NWBFile object.
    """
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
    nwbfile_kwargs: Optional[dict] = None,
    subject_kwargs: Optional[list[dict]] = None,
):
    """
    Export VAME project to NWB format.
    """
    session_names = config.get("session_names")
    if not session_names:
        raise ValueError("No session names provided in the config.")

    if nwbfile_kwargs is None:
        nwbfile_kwargs = {}
    if subject_kwargs is None:
        subject_kwargs = [{}] * len(session_names)

    if len(subject_kwargs) != len(session_names):
        raise ValueError("Number of subject_kwargs must match number of sessions.")

    model_time_window = config["time_window"]
    vame_starting_sample_offset = int(model_time_window / 2)

    for session_name, sub in zip(session_names, subject_kwargs):

        # Load session data
        data_path = (Path(config["project_path"]) / "data" / "processed" / f"{session_name}_processed.nc").resolve()
        ds = load_vame_dataset(ds_path=data_path)
        if ds is None:
            raise ValueError(f"Dataset not found for session: {session_name}")

        keypoints = ds.keypoints.values
        vame_series_kwargs = {}
        if getattr(ds, "fps", None):
            vame_series_kwargs["rate"] = ds.fps
            vame_series_kwargs["starting_time"] = vame_starting_sample_offset / ds.fps
        else:
            vame_series_kwargs["timestamps"] = ds.sel(keypoints=keypoints[0]).time.values

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
            f"{segmentation_algorithms[0]}-{n_clusters}" /
            f"latent_vector_{session_name}.npy"
        ).resolve()
        latent_data = np.load(latent_path)

        for seg in segmentation_algorithms:
            # Base NWB file
            nwbfile = get_base_nwbfile(
                session_name=session_name,
                ds=ds,
                nwbfile_kwargs=nwbfile_kwargs,
                subject_kwargs=sub,
            )
            behavior_pm = nwbfile.processing["behavior"]
            pose_estimation = behavior_pm["PoseEstimation"]

            # Latent space data
            latent_space_series = ndx_vame.LatentSpaceSeries(
                name="LatentSpaceSeries",
                data=latent_data,
                **vame_series_kwargs,
            )

            # Motif data
            motifs_path = (
                Path(config["project_path"]) /
                "results" /
                session_name /
                model_name /
                f"{seg}-{n_clusters}" /
                f"{n_clusters}_{seg}_label_{session_name}.npy"
            ).resolve()
            motif_labels = np.load(motifs_path)
            motif_series = ndx_vame.MotifSeries(
                name="MotifSeries",
                data=motif_labels,
                algorithm=seg,
                latent_space_series=latent_space_series,
                **vame_series_kwargs,
            )

            # Community data
            community_path = (
                Path(config["project_path"]) /
                "results" /
                session_name /
                model_name /
                f"{seg}-{n_clusters}" /
                "community" /
                f"cohort_community_label_{session_name}.npy"
            ).resolve()
            data_communities = np.load(community_path)
            community_series = ndx_vame.CommunitySeries(
                name="CommunitySeries",
                data=data_communities,
                motif_series=motif_series,
                algorithm="hierarchical_clustering",
                **vame_series_kwargs,
            )

            vame_project = ndx_vame.VAMEProject(
                name="VAMEProject",
                pose_estimation=pose_estimation,
                latent_space_series=latent_space_series,
                motif_series=motif_series,
                community_series=community_series,
                vame_config=json.dumps(config),
            )

            behavior_pm.add(vame_project)

            # Save NWB file
            nwbfile_path = (
                Path(config["project_path"]) /
                "results" /
                session_name /
                model_name /
                f"{seg}-{n_clusters}" /
                f"{session_name}.nwb"
            ).resolve()
            with pynwb.NWBHDF5IO(str(nwbfile_path), "w") as io:
                io.write(nwbfile)
