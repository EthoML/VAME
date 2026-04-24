---
sidebar_label: new
title: initialize_project.new
---

#### logger\_config

#### logger

#### init\_new\_project

```python
def init_new_project(project_name: str,
                     poses_estimations: List[str],
                     source_software: Literal["DeepLabCut", "SLEAP",
                                              "LightningPose", "NWB",
                                              "auto"] = "auto",
                     working_directory: str = ".",
                     videos: Optional[List[str]] = None,
                     video_type: str = ".mp4",
                     fps: Optional[float] = None,
                     copy_videos: bool = False,
                     processing_module_key: str = "behavior",
                     pose_estimation_key: str = "PoseEstimation",
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
* **source_software** (`Literal["DeepLabCut", "SLEAP", "LightningPose", "NWB", "auto"], optional`): Source software used for pose estimation. Defaults to ``&quot;auto&quot;``, which
lets movement infer the format from the file extension and contents.
Pass an explicit value (``&quot;DeepLabCut&quot;``, ``&quot;SLEAP&quot;``,
``&quot;LightningPose&quot;``, ``&quot;NWB&quot;``) to override auto-detection.
* **working_directory** (`str, optional`): Working directory. Defaults to &#x27;.&#x27;.
* **video_type** (`str, optional`): Video extension (.mp4 or .avi). Defaults to &#x27;.mp4&#x27;.
* **fps** (`float, optional`): Sampling rate of the videos. If not passed, it will be estimated from the video file. Defaults to None.
* **copy_videos** (`bool, optional`): If True, the videos will be copied to the project directory. If False, symbolic links will be created instead. Defaults to False.
* **processing_module_key** (`str, optional`): Only used when ``source_software=&quot;NWB&quot;``. Name of the NWB processing
module that contains the pose estimation container. Defaults to &quot;behavior&quot;.
* **pose_estimation_key** (`str, optional`): Only used when ``source_software=&quot;NWB&quot;``. Name of the
``ndx_pose.PoseEstimation`` object inside the processing module.
Defaults to &quot;PoseEstimation&quot;.
* **config_kwargs** (`Optional[dict], optional`): Additional configuration parameters. Defaults to None.

**Returns**

* `Tuple[str, dict]`: Tuple containing the path to the config file and the config data.

