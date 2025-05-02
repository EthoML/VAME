from typing import Optional
from datetime import datetime, timezone
from pathlib import Path
import pynwb
import ndx_pose
import ndx_vame

from vame.io import load_vame_dataset


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

    for session_name, sub in zip(session_names, subject_kwargs):
        subject_id = sub.pop("subject_id", session_name)
        subject = pynwb.file.Subject(
            subject_id=subject_id,
            **sub,
        )
        nwbfile = pynwb.NWBFile(
            session_description=nwbfile_kwargs.pop("session_description", "session_description"),
            identifier=nwbfile_kwargs.pop("identifier", session_name),
            session_start_time=nwbfile_kwargs.pop("session_start_time", datetime.now(timezone.utc)),
            subject=subject,
        )

        camera1 = nwbfile.create_device(
            name="camera1",
            description="camera for recording behavior",
            manufacturer="my manufacturer",
        )

        # Load session data
        data_path = (Path(config["project_path"]) / "data" / "processed" / f"{session_name}_processed.nc").resolve()
        ds = load_vame_dataset(ds_path=data_path)
        if ds is None:
            raise ValueError(f"Dataset not found for session: {session_name}")

        # Create pose estimation and skeletons objects
        keypoints = ds.keypoints.values
        pose_estimation_series_kwargs = {}
        if getattr(ds, "fps", None):
            pose_estimation_series_kwargs["fps"] = ds.fps
        else:
            pose_estimation_series_kwargs["timestamps"] = ds.sel(keypoints=keypoint).time.values

        pose_estimation_series_list = []
        for keypoint in keypoints:
            pose_estimation_series_list.append(
                ndx_pose.PoseEstimationSeries(
                    name=keypoint,
                    data=ds.sel(keypoints=keypoint).position.values,
                    confidence=ds.sel(keypoints=keypoint).confidence.values,
                    unit="pixels",
                    **pose_estimation_series_kwargs,
                )
            )

        skeleton = ndx_pose.Skeleton(
            name=f"{subject_id}_skeleton",
            nodes=keypoints,
            subject=subject,
        )

        source_software = getattr(ds, "source_software", "Unknown Software")
        video_path = getattr(ds, "video_path", None)
        pose_estimation = ndx_pose.PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=pose_estimation_series_list,
            description=f"Estimated positions using {source_software}.",
            original_videos=[video_path],
            devices=[camera1],
            source_software=source_software,
            skeleton=skeleton,
        )

        # Create behavior processing module
        behavior_pm = nwbfile.create_processing_module(
            name="behavior",
            description="processed behavioral data",
        )
        behavior_pm.add(pose_estimation)

        # VAME content
        model_name = config.get("model_name")
        n_clusters = config.get("n_clusters")
        segmentation_algorithms = config.get("segmentation_algorithms")
        for seg in segmentation_algorithms:
            data_path = (
                Path(config["project_path"]) /
                "results" /
                session_name /
                model_name /
                f"{seg}-{n_clusters}" /

            ).resolve()