---
sidebar_label: new
title: initialize_project.new
---

#### logger\_config

#### logger

#### init\_new\_project

```python
def init_new_project(project_name: str,
                     videos: List[str],
                     poses_estimations: List[str],
                     source_software: Literal["DeepLabCut", "SLEAP",
                                              "LightningPose"],
                     working_directory: str = ".",
                     video_type: str = ".mp4",
                     fps: int | None = None,
                     copy_videos: bool = False,
                     paths_to_pose_nwb_series_data: Optional[str] = None,
                     config_kwargs: Optional[dict] = None) -> Tuple[str, dict]
```

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

**Parameters**

* **project_name** (`str`): Project name.
* **videos** (`List[str]`): List of videos paths to be used in the project. E.g. [&#x27;./sample_data/Session001.mp4&#x27;]
* **poses_estimations** (`List[str]`): List of pose estimation files paths to be used in the project. E.g. [&#x27;./sample_data/pose estimation/Session001.csv&#x27;]
* **source_software** (`Literal["DeepLabCut", "SLEAP", "LightningPose"]`): Source software used for pose estimation.
* **working_directory** (`str, optional`): Working directory. Defaults to &#x27;.&#x27;.
* **video_type** (`str, optional`): Video extension (.mp4 or .avi). Defaults to &#x27;.mp4&#x27;.
* **fps** (`int, optional`): Sampling rate of the videos. If not passed, it will be estimated from the video file. Defaults to None.
* **copy_videos** (`bool, optional`): If True, the videos will be copied to the project directory. If False, symbolic links will be created instead. Defaults to False.
* **paths_to_pose_nwb_series_data** (`Optional[str], optional`): List of paths to the pose series data in nwb files. Defaults to None.
* **config_kwargs** (`Optional[dict], optional`): Additional configuration parameters. Defaults to None.

**Returns**

* `Tuple[str, dict]`: Tuple containing the path to the config file and the config data.

