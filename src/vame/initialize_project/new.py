from typing import List, Optional, Literal, Tuple
from datetime import datetime, timezone
from pathlib import Path
import shutil
import json
import os

from vame.schemas.project import ProjectSchema, PoseEstimationFiletype
from vame.schemas.states import VAMEPipelineStatesSchema
from vame.logging.logger import VameLogger
from vame.util.auxiliary import write_config, read_config
from vame.video.video import get_video_frame_rate
from vame.io.load_poses import load_pose_estimation


logger_config = VameLogger(__name__)
logger = logger_config.logger


def init_new_project(
    project_name: str,
    videos: List[str],
    poses_estimations: List[str],
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
    working_directory: str = ".",
    video_type: str = ".mp4",
    fps: int | None = None,
    copy_videos: bool = False,
    paths_to_pose_nwb_series_data: Optional[str] = None,
    config_kwargs: Optional[dict] = None,
) -> Tuple[str, dict]:
    """
    Creates a new VAME project with the given parameters.
    A VAME project is a directory with the following structure:
    - project_name/
        - data/
            - raw/
                - session1.mp4
                - session1.nc
                - session2.mp4
                - session2.nc
                - ...
            - processed/
                - session1_processed.nc
                - session2_processed.nc
                - ...
        - model/
            - pretrained_model/
        - results/
            - video1/
            - video2/
            - ...
        - states/
            - states.json
        - config.yaml

    Parameters
    ----------
    project_name : str
        Project name.
    videos : List[str]
        List of videos paths to be used in the project. E.g. ['./sample_data/Session001.mp4']
    poses_estimations : List[str]
        List of pose estimation files paths to be used in the project. E.g. ['./sample_data/pose estimation/Session001.csv']
    source_software : Literal["DeepLabCut", "SLEAP", "LightningPose"]
        Source software used for pose estimation.
    working_directory : str, optional
        Working directory. Defaults to '.'.
    video_type : str, optional
        Video extension (.mp4 or .avi). Defaults to '.mp4'.
    fps : int, optional
        Sampling rate of the videos. If not passed, it will be estimated from the video file. Defaults to None.
    copy_videos : bool, optional
        If True, the videos will be copied to the project directory. If False, symbolic links will be created instead. Defaults to False.
    paths_to_pose_nwb_series_data : Optional[str], optional
        List of paths to the pose series data in nwb files. Defaults to None.
    config_kwargs : Optional[dict], optional
        Additional configuration parameters. Defaults to None.

    Returns
    -------
    Tuple[str, dict]
        Tuple containing the path to the config file and the config data.
    """
    creation_datetime = datetime.now(timezone.utc).isoformat(timespec="seconds")
    project_path = Path(working_directory).resolve() / project_name
    if project_path.exists():
        logger.info('Project "{}" already exists!'.format(project_path))
        projconfigfile = os.path.join(str(project_path), "config.yaml")
        return projconfigfile, read_config(projconfigfile)

    data_path = project_path / "data"
    data_raw_path = data_path / "raw"
    data_processed_path = data_path / "processed"
    results_path = project_path / "results"
    model_path = project_path / "model"
    model_evaluate_path = model_path / "evaluate"
    model_pretrained_path = model_path / "pretrained_model"
    for p in [
        data_path,
        data_raw_path,
        data_processed_path,
        results_path,
        model_path,
        model_pretrained_path,
        model_evaluate_path,
    ]:
        p.mkdir(parents=True)
        logger.info('Created "{}"'.format(p))

    vids = []
    for i in videos:
        # Check if it is a folder
        if os.path.isdir(i):
            vids_in_dir = [os.path.join(i, vp) for vp in os.listdir(i) if video_type in vp]
            vids = vids + vids_in_dir
            if len(vids_in_dir) == 0:
                logger.info(f"No videos found in {i}")
                logger.info(f"Perhaps change the video_type, which is currently set to: {video_type}")
            else:
                videos = vids
                logger.info(f"{len(vids_in_dir)} videos from the directory {i} were added to the project.")
        else:
            if os.path.isfile(i):
                vids = vids + [i]
            videos = vids

    pose_estimations_paths = []
    for pose_estimation_path in poses_estimations:
        if os.path.isdir(pose_estimation_path):
            pose_estimation_files = [
                os.path.join(pose_estimation_path, p)
                for p in os.listdir(pose_estimation_path)
                if ".csv" in p or ".nwb" in p
            ]
            pose_estimations_paths.extend(pose_estimation_files)
        else:
            pose_estimations_paths.append(pose_estimation_path)

    # if not all([p.endswith(".csv") for p in pose_estimations_paths]) and not all(
    #     [p.endswith(".nwb") for p in pose_estimations_paths]
    # ):
    #     logger.error(
    #         "All pose estimation files must be in the same format. Either .csv or .nwb"
    #     )
    #     shutil.rmtree(str(project_path))
    #     raise ValueError(
    #         "All pose estimation files must be in the same format. Either .csv or .nwb"
    #     )

    pose_estimation_filetype = pose_estimations_paths[0].split(".")[-1]
    if (
        pose_estimation_filetype == PoseEstimationFiletype.nwb.value
        and paths_to_pose_nwb_series_data
        and len(paths_to_pose_nwb_series_data) != len(pose_estimations_paths)
    ):
        logger.error(
            "If the pose estimation file is in nwb format, you must provide the path to the pose series data in nwb files."
        )
        shutil.rmtree(str(project_path))
        raise ValueError(
            "If the pose estimation file is in nwb format, you must provide the path to the pose series data for each nwb file."
        )

    # Session names
    videos_paths = [Path(vp).resolve() for vp in videos]
    session_names = []
    for s in videos_paths:
        session_names.append(s.stem)

    # # Creates directories under project/data/processed/
    # dirs_processed_data = [data_processed_path / Path(i.stem) for i in videos_paths]
    # for p in dirs_processed_data:
    #     p.mkdir(parents=True, exist_ok=True)

    # Creates directories under project/results/
    dirs_results = [results_path / Path(i.stem) for i in videos_paths]
    for p in dirs_results:
        p.mkdir(parents=True, exist_ok=True)

    logger.info("Copying / linking the video files... \n")
    destinations = [data_raw_path / vp.name for vp in videos_paths]
    for src, dst in zip(videos_paths, destinations):
        if copy_videos:
            logger.info(f"Copying {src} to {dst}")
            shutil.copy(os.fspath(src), os.fspath(dst))
        else:
            logger.info(f"Creating symbolic link from {src} to {dst}")
            os.symlink(os.fspath(src), os.fspath(dst))

    if fps is None:
        fps = get_video_frame_rate(str(videos_paths[0]))

    logger.info("Copying pose estimation raw data...\n")
    num_features_list = []
    for pes_path, video_path in zip(pose_estimations_paths, videos_paths):
        ds = load_pose_estimation(
            pose_estimation_file=pes_path,
            video_file=video_path,
            fps=fps,
            source_software=source_software,
        )
        output_name = data_raw_path / Path(video_path).stem
        ds.to_netcdf(
            path=f"{output_name}.nc",
            engine="scipy",
        )
        num_features_list.append(ds.space.shape[0] * ds.keypoints.shape[0])

        output_processed_name = data_processed_path / Path(video_path).stem
        ds.to_netcdf(
            path=f"{output_processed_name}_processed.nc",
            engine="scipy",
        )

    unique_num_features = list(set(num_features_list))
    if len(unique_num_features) > 1:
        raise ValueError("All pose estimation files must have the same number of features.")

    if config_kwargs is None:
        config_kwargs = {}
    config_kwargs["num_features"] = unique_num_features[0]

    new_project = ProjectSchema(
        project_name=project_name,
        creation_datetime=creation_datetime,
        project_path=str(project_path),
        session_names=session_names,
        pose_estimation_filetype=pose_estimation_filetype,
        paths_to_pose_nwb_series_data=paths_to_pose_nwb_series_data,
        **config_kwargs,
    )
    # Write dictionary to yaml  config file
    cfg_data = new_project.model_dump()
    projconfigfile = os.path.join(str(project_path), "config.yaml")
    write_config(projconfigfile, cfg_data)

    vame_pipeline_default_schema = VAMEPipelineStatesSchema()
    vame_pipeline_default_schema_path = Path(project_path) / "states" / "states.json"
    if not vame_pipeline_default_schema_path.parent.exists():
        vame_pipeline_default_schema_path.parent.mkdir(parents=True)
    with open(vame_pipeline_default_schema_path, "w") as f:
        json.dump(vame_pipeline_default_schema.model_dump(), f, indent=4)

    logger.info(f"A VAME project has been created at {project_path}")

    return projconfigfile, cfg_data
