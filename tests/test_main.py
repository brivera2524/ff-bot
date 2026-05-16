import argparse
from pathlib import Path

import cv2
import numpy as np
import pytest

import main


def test_parse_region_accepts_valid_coordinates():
    assert main.parse_region("10, 20, 300, 40") == (10, 20, 300, 40)


@pytest.mark.parametrize("value", ["10,20,30", "10,20,width,40", "10,20,0,40"])
def test_parse_region_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        main.parse_region(value)


def test_parse_args_accepts_custom_lane_keys():
    args = main.parse_args(["--keys", "asdfg", "--delay", "0.05"])

    assert args.keys == "asdfg"
    assert args.delay == 0.05


def test_parse_args_rejects_wrong_number_of_lane_keys():
    with pytest.raises(SystemExit):
        main.parse_args(["--keys", "asdf"])


def test_parse_args_rejects_duplicate_lane_keys():
    with pytest.raises(SystemExit):
        main.parse_args(["--keys", "aassd"])


def test_parse_args_rejects_control_keys_that_overlap_lane_keys():
    with pytest.raises(SystemExit):
        main.parse_args(["--keys", "asdfg", "--pause-key", "a"])


def test_parse_args_rejects_matching_pause_and_quit_keys():
    with pytest.raises(SystemExit):
        main.parse_args(["--pause-key", "x", "--quit-key", "x"])


def test_build_lane_states_uses_configured_keys():
    assert main.build_lane_states("asdfg") == {
        "a": 1,
        "s": 1,
        "d": 1,
        "f": 1,
        "g": 1,
    }


def test_load_press_template_raises_clear_error_for_missing_file():
    with pytest.raises(FileNotFoundError, match="Missing template image"):
        main.load_press_template(Path("missing-template-for-test.png"))


def test_load_press_template_returns_grayscale_image():
    image_path = Path(__file__).parent / "_generated_template.png"
    color_image = np.zeros((4, 5, 3), dtype=np.uint8)
    color_image[:, :, 1] = 255
    try:
        cv2.imwrite(str(image_path), color_image)

        loaded = main.load_press_template(image_path)

        assert loaded.shape == (4, 5)
        assert loaded.dtype == np.uint8
    finally:
        image_path.unlink(missing_ok=True)
