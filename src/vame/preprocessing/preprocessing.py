import os
from pathlib import Path
from typing import Callable, List, Tuple

from vame.io.load_poses import load_vame_dataset
from vame.logging.logger import VameLogger
from vame.preprocessing import alignment, cleaning, scaling
from vame.preprocessing import filter as filtering
from vame.preprocessing.cleaning import lowconf_clean_dataset, outlier_clean_dataset
from vame.preprocessing.alignment import egocentrically_align_dataset
from vame.preprocessing.filter import savgol_filter_dataset
from vame.preprocessing.scaling import rescale_dataset
from vame.schemas.states import save_state, PreprocessingFunctionSchema
from vame.util.auxiliary import update_config


logger_config = VameLogger(__name__)
logger = logger_config.logger


@save_state(model=PreprocessingFunctionSchema)
def preprocessing(
    config: dict,
    centered_reference_keypoint: str,
    orientation_reference_keypoint: str,
    run_lowconf_cleaning: bool = True,
    run_egocentric_alignment: bool = True,
    run_outlier_cleaning: bool = True,
    run_savgol_filtering: bool = True,
    run_rescaling: bool = True,
    save_logs: bool = True,
) -> str:
    """
    Preprocess the data by:
        - Cleaning low confidence data points
        - Egocentric alignment
        - Outlier cleaning using IQR
        - Savitzky-Golay filtering
        - Rescaling

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    centered_reference_keypoint : str, optional
        Keypoint to use as centered reference.
    orientation_reference_keypoint : str, optional
        Keypoint to use as orientation reference.
    run_lowconf_cleaning : bool, optional
        Whether to run low confidence cleaning.
    run_egocentric_alignment : bool, optional
        Whether to run egocentric alignment.
    run_outlier_cleaning : bool, optional
        Whether to run outlier cleaning.
    run_savgol_filtering : bool, optional
        Whether to run Savitzky-Golay filtering.
    run_rescaling : bool, optional
        Whether to run rescaling. Defaults to True.
    save_logs : bool, optional
        Whether to save logs.

    Returns
    -------
    variable name of the last-executed preprocessing step output, also saved
    to the config as ``preprocessed_variable``
    """
    if save_logs:
        log_path = Path(config["project_path"]) / "logs" / "preprocessing.log"
        for step_logger_config in (
            logger_config,
            cleaning.logger_config,
            alignment.logger_config,
            filtering.logger_config,
            scaling.logger_config,
        ):
            step_logger_config.add_file_handler(str(log_path))

    steps = []
    latest_output = "position"

    # Low-confidence cleaning
    if run_lowconf_cleaning:
        logger.info(f"Cleaning low confidence data points. Confidence threshold: {config['pose_confidence']}")
        steps.append(
            (
                lowconf_clean_dataset,
                {
                    "pose_confidence": config["pose_confidence"],
                    "read_from_variable": latest_output,
                    "save_to_variable": "position_cleaned_lowconf",
                },
            )
        )
        latest_output = "position_cleaned_lowconf"

    # Egocentric alignment
    if run_egocentric_alignment:
        logger.info(
            "Egocentrically aligning and centering with references: "
            f"{centered_reference_keypoint} and {orientation_reference_keypoint}"
        )
        steps.append(
            (
                egocentrically_align_dataset,
                {
                    "centered_reference_keypoint": centered_reference_keypoint,
                    "orientation_reference_keypoint": orientation_reference_keypoint,
                    "read_from_variable": latest_output,
                    "save_to_variable": "position_egocentric_aligned",
                },
            )
        )
        latest_output = "position_egocentric_aligned"

    # Outlier cleaning
    if run_outlier_cleaning:
        logger.info("Cleaning outliers using IQR method...")
        steps.append(
            (
                outlier_clean_dataset,
                {
                    "robust": config["robust"],
                    "iqr_factor": config["iqr_factor"],
                    "read_from_variable": latest_output,
                    "save_to_variable": "position_processed",
                },
            )
        )
        latest_output = "position_processed"

    # Savgol filtering
    if run_savgol_filtering:
        logger.info("Applying Savitzky-Golay filter...")
        steps.append(
            (
                savgol_filter_dataset,
                {
                    "savgol_length": config["savgol_length"],
                    "savgol_order": config["savgol_order"],
                    "read_from_variable": latest_output,
                    "save_to_variable": "position_processed",
                },
            )
        )
        latest_output = "position_processed"

    # Rescaling
    if run_rescaling:
        logger.info("Rescaling...")
        steps.append(
            (
                rescale_dataset,
                {
                    "read_from_variable": latest_output,
                    "save_to_variable": "position_scaled",
                },
            )
        )
        latest_output = "position_scaled"

    if steps:
        processed_dir = Path(config["project_path"]) / "data" / "processed"
        for session in config["session_names"]:
            logger.info(f"Session: {session}")
            preprocess_session(file_path=str(processed_dir / f"{session}_processed.nc"), steps=steps)

    # create_trainset reads this by default
    update_config(config=config, config_update={"preprocessed_variable": latest_output})

    return latest_output


def preprocess_session(file_path: str, steps: List[Tuple[Callable, dict]]) -> None:
    """
    Run preprocessing steps on one session's processed file: read once, apply the
    steps in order, and replace the file once.

    Parameters
    ----------
    file_path : str
        Path to the session's processed netCDF file.
    steps : list of (function, kwargs)
        Dataset functions to apply in order, each called as ``function(ds=ds, **kwargs)``.

    Returns
    -------
    None
    """
    ds = load_vame_dataset(ds_path=file_path)
    for step_function, step_kwargs in steps:
        step_function(ds=ds, **step_kwargs)

    # Write next to the original and swap, so an interrupted run never leaves a partial file
    tmp_path = f"{file_path}.tmp"
    ds.to_netcdf(path=tmp_path, engine="netcdf4")
    os.replace(tmp_path, file_path)

