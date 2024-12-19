from vame.logging.logger import VameLogger
from vame.preprocessing.cleaning import lowconf_cleaning, outlier_cleaning
from vame.preprocessing.alignment import egocentrically_align_and_center
from vame.preprocessing.filter import savgol_filtering
from vame.preprocessing.visualization import (
    visualize_preprocessing_scatter,
    visualize_preprocessing_timeseries,
)


logger_config = VameLogger(__name__)
logger = logger_config.logger


def preprocessing(
    config: dict,
    centered_reference_keypoint: str = "snout",
    orientation_reference_keypoint: str = "tailbase",
    save_logs: bool = False,
):

    # Low-confidence cleaning
    logger.info("Cleaning low confidence data points...")
    lowconf_cleaning(config=config)

    # Egocentric alignment
    logger.info("Egocentrically aligning and centering...")
    egocentrically_align_and_center(
        config=config,
        centered_reference_keypoint=centered_reference_keypoint,
        orientation_reference_keypoint=orientation_reference_keypoint,
    )
    visualize_preprocessing_scatter(
        config,
        save_to_file=True,
    )

    # Outlier cleaning
    logger.info("Cleaning outliers...")
    outlier_cleaning(config=config)

    # Savgol filtering
    logger.info("Applying Savitzky-Golay filter...")
    savgol_filtering(config=config)

    visualize_preprocessing_timeseries(
        config,
        save_to_file=True,
    )
