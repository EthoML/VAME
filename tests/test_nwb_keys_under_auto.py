from pathlib import Path
from unittest.mock import patch

import pytest

from vame.io.load_poses import load_pose_estimation

NWB_FILE = Path(__file__).parent / "tests_project_sample_nwb" / "cropped_video.nwb"

pytestmark = pytest.mark.skipif(not NWB_FILE.exists(), reason="NWB sample missing")


def _captured_kwargs(source_software):
    with patch("vame.io.load_poses.load_dataset") as mock:
        try:
            load_pose_estimation(
                pose_estimation_file=NWB_FILE,
                source_software=source_software,
                processing_module_key="my_module",
                pose_estimation_key="my_poses",
            )
        except Exception:
            pass
        return mock.call_args.kwargs if mock.call_args else {}


def test_keys_forwarded_for_explicit_nwb():
    kwargs = _captured_kwargs("NWB")
    assert kwargs.get("processing_module_key") == "my_module"
    assert kwargs.get("pose_estimation_key") == "my_poses"


def test_keys_forwarded_under_auto():
    kwargs = _captured_kwargs("auto")
    assert kwargs.get("processing_module_key") == "my_module"
    assert kwargs.get("pose_estimation_key") == "my_poses"


def test_auto_is_resolved_before_dispatch():
    assert _captured_kwargs("auto").get("source_software") == "NWB"
