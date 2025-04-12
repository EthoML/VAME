from functools import wraps
from pydantic import BaseModel, Field
from typing import Optional, Dict
from pathlib import Path
import json
from enum import Enum
from vame.schemas.project import SegmentationAlgorithms


class StatesEnum(str, Enum):
    success = "success"
    failed = "failed"
    running = "running"
    aborted = "aborted"


class GenerativeModelModeEnum(str, Enum):
    sampling = "sampling"
    reconstruction = "reconstruction"
    centers = "centers"
    motifs = "motifs"


class BaseStateSchema(BaseModel):
    config: dict = Field(title="Configuration dictionary")
    execution_state: StatesEnum | None = Field(
        title="Method execution state",
        default=None,
    )


class PoseToNumpyFunctionSchema(BaseStateSchema):
    ...


class CreateTrainsetFunctionSchema(BaseStateSchema):
    ...


class TrainModelFunctionSchema(BaseStateSchema):
    ...


class EvaluateModelFunctionSchema(BaseStateSchema):
    use_snapshots: bool = Field(
        title="Use snapshots",
        default=False,
    )


class SegmentSessionFunctionSchema(BaseStateSchema):
    ...


class MotifVideosFunctionSchema(BaseStateSchema):
    video_type: str = Field(
        title="Type of video",
        default=".mp4",
    )
    segmentation_algorithm: SegmentationAlgorithms = Field(title="Segmentation algorithm")
    output_video_type: str = Field(
        title="Type of output video",
        default=".mp4",
    )


class CommunityFunctionSchema(BaseStateSchema):
    cohort: bool = Field(title="Cohort", default=True)
    segmentation_algorithm: SegmentationAlgorithms = Field(title="Segmentation algorithm")
    cut_tree: int | None = Field(
        title="Cut tree",
        default=None,
    )


class CommunityVideosFunctionSchema(BaseStateSchema):
    segmentation_algorithm: SegmentationAlgorithms = Field(title="Segmentation algorithm")
    cohort: bool = Field(title="Cohort", default=True)
    video_type: str = Field(
        title="Type of video",
        default=".mp4",
    )
    output_video_type: str = Field(
        title="Type of output video",
        default=".mp4",
    )


class VisualizeUmapFunctionSchema(BaseStateSchema):
    segmentation_algorithm: SegmentationAlgorithms = Field(title="Segmentation algorithm")
    label: Optional[str] = Field(
        title="Type of labels to visualize",
        default=None,
    )


class PreprocessingFunctionSchema(BaseStateSchema):
    centered_reference_keypoint: str = Field(
        title="Keypoint to use as centered reference",
    )
    orientation_reference_keypoint: str = Field(
        title="Keypoint to use as orientation reference",
    )
    run_lowconf_cleaning: bool = Field(
        title="Whether to run low confidence cleaning",
        default=True,
    )
    run_egocentric_alignment: bool = Field(
        title="Whether to run egocentric alignment",
        default=True,
    )
    run_outlier_cleaning: bool = Field(
        title="Whether to run outlier cleaning",
        default=True,
    )
    run_savgol_filtering: bool = Field(
        title="Whether to run Savitzky-Golay filtering",
        default=True,
    )
    run_rescaling: bool = Field(
        title="Whether to run rescaling",
        default=False,
    )
    save_logs: bool = Field(
        title="Whether to save logs",
        default=False,
    )


class GenerativeModelFunctionSchema(BaseStateSchema):
    segmentation_algorithm: SegmentationAlgorithms = Field(title="Segmentation algorithm")
    mode: GenerativeModelModeEnum = Field(
        title="Mode for generating samples",
        default=GenerativeModelModeEnum.sampling,
    )


class VAMEPipelineStatesSchema(BaseModel):
    preprocessing: Optional[PreprocessingFunctionSchema | Dict] = Field(
        title="Preprocessing",
        default={},
    )
    pose_to_numpy: Optional[PoseToNumpyFunctionSchema | Dict] = Field(
        title="CSV to numpy",
        default={},
    )
    create_trainset: Optional[CreateTrainsetFunctionSchema | Dict] = Field(
        title="Create trainset",
        default={},
    )
    train_model: Optional[TrainModelFunctionSchema | Dict] = Field(
        title="Train model",
        default={},
    )
    evaluate_model: Optional[EvaluateModelFunctionSchema | Dict] = Field(
        title="Evaluate model",
        default={},
    )
    segment_session: Optional[SegmentSessionFunctionSchema | Dict] = Field(
        title="Segment session",
        default={},
    )
    motif_videos: Optional[MotifVideosFunctionSchema | Dict] = Field(
        title="Motif videos",
        default={},
    )
    community: Optional[CommunityFunctionSchema | Dict] = Field(
        title="Community",
        default={},
    )
    community_videos: Optional[CommunityVideosFunctionSchema | Dict] = Field(
        title="Community videos",
        default={},
    )
    visualize_umap: Optional[VisualizeUmapFunctionSchema | Dict] = Field(
        title="Visualization",
        default={},
    )
    generative_model: Optional[GenerativeModelFunctionSchema | Dict] = Field(
        title="Generative model",
        default={},
    )


def _save_state(model: BaseStateSchema, function_name: str, state: StatesEnum) -> None:
    """
    Save the state of the function to the project states json file.
    """
    states_file_path = Path(model.config["project_path"]) / "states" / "states.json"
    with open(states_file_path, "r") as f:
        states = json.load(f)

    pipeline_states = VAMEPipelineStatesSchema(**states)
    model.execution_state = state
    setattr(pipeline_states, function_name, model.model_dump())

    with open(states_file_path, "w") as f:
        json.dump(pipeline_states.model_dump(), f, indent=4)


def save_state(model: BaseModel):
    """
    Decorator responsible for validating function arguments using pydantic and
    saving the state of the called function to the project states json file.
    """

    def decorator(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create an instance of the Pydantic model using provided args and kwargs
            function_name = func.__name__
            attribute_names = list(model.model_fields.keys())

            kwargs_dict = {}
            for attr in attribute_names:
                if attr == "execution_state":
                    kwargs_dict[attr] = "running"
                    continue
                kwargs_dict[attr] = kwargs.get(attr, model.model_fields[attr].default)

            # Override with positional arguments
            for i, arg in enumerate(args):
                kwargs_dict[attribute_names[i]] = arg
            # Validate function args and kwargs using the Pydantic model.
            kwargs_model = model(**kwargs_dict)
            _save_state(kwargs_model, function_name, state=StatesEnum.running)
            try:
                func_output = func(*args, **kwargs)
                _save_state(kwargs_model, function_name, state=StatesEnum.success)
                return func_output
            except Exception as e:
                _save_state(kwargs_model, function_name, state=StatesEnum.failed)
                raise e
            except KeyboardInterrupt as e:
                _save_state(kwargs_model, function_name, state=StatesEnum.aborted)
                raise e

        return wrapper

    return decorator
