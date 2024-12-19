from vame.logging.logger import VameLogger
from vame.preprocessing.cleaning import lowconf_cleaning
from vame.preprocessing.alignment import egocentrically_align_and_center


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

    # outlier_cleaning(config=config)

    # savgol_filtering(config=config)
