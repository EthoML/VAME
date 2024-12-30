from typing import List, Optional, Literal
from pathlib import Path
import xarray as xr

import vame
from vame.util.auxiliary import read_config, read_states
from vame.io.load_poses import load_vame_dataset
from vame.visualization.umap import visualize_umap
from vame.visualization.preprocessing import (
    visualize_preprocessing_scatter,
    visualize_preprocessing_timeseries,
)
from vame.logging.logger import VameLogger


logger_config = VameLogger(__name__)
logger = logger_config.logger


class VAMEPipeline:
    """VAME pipeline class."""

    def __init__(
        self,
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
    ) -> None:
        """
        Initializes the VAME pipeline.

        Parameters
        ----------
        project_name : str
            Project name.
        videos : List[str]
            List of video files.
        poses_estimations : List[str]
            List of pose estimation files.
        source_software : Literal["DeepLabCut", "SLEAP", "LightningPose"]
            Source software used for pose estimation.
        working_directory : str, optional
            Working directory, by default ".".
        video_type : str, optional
            Video file type, by default ".mp4".
        fps : int, optional
            Sampling rate of the videos. If not passed, it will be estimated from the video file. By default None.
        copy_videos : bool, optional
            Copy videos, by default False.
        paths_to_pose_nwb_series_data : Optional[str], optional
            Path to pose NWB series data, by default None.
        config_kwargs : Optional[dict], optional
            Additional configuration keyword arguments, by default None.

        Returns
        -------
        None
        """
        self.config_path, self.config = vame.init_new_project(
            project_name=project_name,
            videos=videos,
            poses_estimations=poses_estimations,
            source_software=source_software,
            working_directory=working_directory,
            video_type=video_type,
            fps=fps,
            copy_videos=copy_videos,
            paths_to_pose_nwb_series_data=paths_to_pose_nwb_series_data,
            config_kwargs=config_kwargs,
        )
        self.config = read_config(self.config_path)

    def get_states(self, summary: bool = True) -> dict:
        """
        Returns the pipeline states.

        Returns
        -------
        dict
            Pipeline states.
        """
        states = read_states(self.config)
        if summary and states:
            logger.info("Pipeline states:")
            for key, value in states.items():
                logger.info(f"{key}: {value.get('execution_state', 'Not executed')}")
        return states

    def get_sessions(self) -> List[str]:
        """
        Returns a list of session names.

        Returns
        -------
        List[str]
            Session names.
        """
        return self.config["session_names"]

    def get_raw_datasets(self) -> xr.Dataset:
        """
        Returns a xarray dataset which combines all the raw data from the project.

        Returns
        -------
        dss : xarray.Dataset
            Combined raw dataset.
        """
        sessions = self.get_sessions()
        datasets = list()
        attributes = list()
        for session in sessions:
            ds_path = Path(self.config["project_path"]) / "data" / "raw" / f"{session}.nc"
            ds = load_vame_dataset(ds_path=ds_path)
            ds = ds.expand_dims({"session": [session]})
            datasets.append(ds)
            attributes.append(ds.attrs)
        dss = xr.concat(datasets, dim="session")
        dss_attrs = {}
        for d in attributes:
            for key, value in d.items():
                dss_attrs.setdefault(key, []).append(value)
        for key, values in dss_attrs.items():
            unique_values = unique_in_order(values)  # Maintain order of unique values
            dss_attrs[key] = unique_values[0] if len(unique_values) == 1 else unique_values
        for key, value in dss_attrs.items():
            dss.attrs[key] = value
        return dss

    def preprocessing(
        self,
        centered_reference_keypoint: str = "snout",
        orientation_reference_keypoint: str = "tailbase",
    ) -> None:
        """
        Preprocesses the data.

        Parameters
        ----------
        centered_reference_keypoint : str, optional
            Key point to center the data, by default "snout".
        orientation_reference_keypoint : str, optional
            Key point to orient the data, by default "tailbase".

        Returns
        -------
        None
        """
        self.centered_reference_keypoint = centered_reference_keypoint
        self.orientation_reference_keypoint = orientation_reference_keypoint
        vame.preprocessing(
            config=self.config,
            centered_reference_keypoint=centered_reference_keypoint,
            orientation_reference_keypoint=orientation_reference_keypoint,
        )

    def create_training_set(self) -> None:
        """
        Creates the training set.

        Returns
        -------
        None
        """
        vame.create_trainset(config=self.config)

    def train_model(self) -> None:
        """
        Trains the model.

        Returns
        -------
        None
        """
        vame.train_model(config=self.config)

    def evaluate_model(self) -> None:
        """
        Evaluates the model.

        Returns
        -------
        None
        """
        vame.evaluate_model(config=self.config)

    def run_segmentation(self) -> None:
        """
        Runs the pose estimation segmentation into motifs.

        Returns
        -------
        None
        """
        vame.segment_session(config=self.config)

    def run_community_clustering(self) -> None:
        """
        Runs the community clustering.

        Returns
        -------
        None
        """
        vame.community(
            config=self.config,
            segmentation_algorithm="hmm",
            cohort=True,
            cut_tree=2,
        )

    def generate_motif_videos(
        self,
        video_type: str = ".mp4",
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
    ) -> None:
        """
        Generates motif videos.

        Parameters
        ----------
        video_type : str, optional
            Video type, by default ".mp4".
        segmentation_algorithm : Literal["hmm", "kmeans"], optional
            Segmentation algorithm, by default "hmm".

        Returns
        -------
        None
        """
        vame.motif_videos(
            config=self.config,
            video_type=video_type,
            segmentation_algorithm=segmentation_algorithm,
        )

    def generate_community_videos(
        self,
        video_type: str = ".mp4",
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
    ) -> None:
        """
        Generates community videos.

        Parameters
        ----------
        video_type : str, optional
            Video type, by default ".mp4".
        segmentation_algorithm : Literal["hmm", "kmeans"], optional
            Segmentation algorithm, by default "hmm".

        Returns
        -------
        None
        """
        vame.community_videos(
            config=self.config,
            video_type=video_type,
            segmentation_algorithm=segmentation_algorithm,
        )

    def generate_videos(
        self,
        video_type: str = ".mp4",
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
    ) -> None:
        """
        Generates motif and community videos.

        Parameters
        ----------
        video_type : str, optional
            Video type, by default ".mp4".
        segmentation_algorithm : Literal["hmm", "kmeans"], optional
            Segmentation algorithm, by default "hmm".

        Returns
        -------
        None
        """
        self.generate_motif_videos(
            video_type=video_type,
            segmentation_algorithm=segmentation_algorithm,
        )
        self.generate_community_videos(
            video_type=video_type,
            segmentation_algorithm=segmentation_algorithm,
        )

    def visualize_prepocessing(
        self,
        scatter: bool = True,
        timeseries: bool = True,
        show_figure: bool = False,
        save_to_file: bool = True,
    ) -> None:
        """
        Visualizes the preprocessing results.

        Parameters
        ----------
        scatter : bool, optional
            Visualize scatter plot, by default True.
        timeseries : bool, optional
            Visualize timeseries plot, by default True.
        show_figure : bool, optional
            Show the figure, by default False.
        save_to_file : bool, optional
            Save the figure to file, by default True.

        Returns
        -------
        None
        """
        if scatter:
            visualize_preprocessing_scatter(
                config=self.config,
                show_figure=show_figure,
                save_to_file=save_to_file,
            )
        if timeseries:
            visualize_preprocessing_timeseries(
                config=self.config,
                show_figure=show_figure,
                save_to_file=save_to_file,
            )

    def visualize_umap(
        self,
        label: Literal["community", "motif"] = "community",
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
        show_figure: bool = False,
        save_to_file: bool = True,
    ) -> None:
        """
        Visualizes the UMAP plot.

        Parameters
        ----------
        label : Literal["community", "motif"], optional
            Label to visualize, by default "community".
        segmentation_algorithm : Literal["hmm", "kmeans"], optional
            Segmentation algorithm, by default "hmm".

        Returns
        -------
        None
        """
        visualize_umap(
            config=self.config,
            label=label,
            segmentation_algorithm=segmentation_algorithm,
        )

    def visualize_results(
        self,
        label: Literal["community", "motif"] = "community",
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
        show_figure: bool = False,
        save_to_file: bool = True,
    ) -> None:
        """
        Visualize results.

        Parameters
        ----------
        label : Literal["community", "motif"], optional
            Label to visualize, by default "community".
        segmentation_algorithm : Literal["hmm", "kmeans"], optional
            Segmentation algorithm, by default "hmm".
        show_figure : bool, optional
            Show the figure, by default False.
        save_to_file : bool, optional
            Save the figure to file, by default True.

        Returns
        -------
        None
        """
        self.visualize_prepocessing(
            scatter=True,
            timeseries=True,
            show_figure=show_figure,
            save_to_file=save_to_file,
        )
        self.visualize_umap(
            label=label,
            segmentation_algorithm=segmentation_algorithm,
            show_figure=show_figure,
            save_to_file=save_to_file,
        )

    def report(
        self,
        segmentation_algorithm: Literal["hmm", "kmeans"] = "hmm",
    ) -> None:
        """
        Generates the project report.

        Parameters
        ----------
        segmentation_algorithm : Literal["hmm", "kmeans"], optional
            Segmentation algorithm, by default "hmm".

        Returns
        -------
        None
        """
        vame.report(
            config=self.config,
            segmentation_algorithm=segmentation_algorithm,
        )

    def run_pipeline(
        self,
        from_step: int = 0,
        preprocessing_kwargs: dict = {},
    ) -> None:
        """
        Runs the pipeline.

        Parameters
        ----------
        from_step : int, optional
            Start from step, by default 0.
        preprocessing_kwargs : dict, optional
            Preprocessing keyword arguments, by default {}.

        Returns
        -------
        None
        """
        if from_step == 0:
            self.preprocessing(**preprocessing_kwargs)
        if from_step <= 1:
            self.create_training_set()
        if from_step <= 2:
            self.train_model()
        if from_step <= 3:
            self.evaluate_model()
        if from_step <= 4:
            self.run_segmentation()
        if from_step <= 5:
            self.run_community_clustering()
        if from_step <= 6:
            self.report()


def unique_in_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
