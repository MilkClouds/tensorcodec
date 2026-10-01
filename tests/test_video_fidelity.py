"""Precision and orientation checks against independent FFmpeg output."""

import numpy as np
import pytest

from tensorcodec.decoders import VideoDecoder
from tests.utils import ffmpeg_rgb, probe_stream, run_ffmpeg


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, "auto"])
@pytest.mark.parametrize("order", ["NHWC", "NCHW"])
def test_precision_and_hdr(precision_video, dtype, order):
    path, depth, transfer, color_range = precision_video
    expected_dtype = np.float32 if dtype == "auto" and depth > 8 else np.uint8 if dtype == "auto" else dtype
    with VideoDecoder(path, output_dtype=dtype, dimension_order=order) as decoder:
        assert decoder.metadata.bit_depth == depth
        assert decoder.metadata.color_range == color_range
        assert decoder.metadata.color_transfer_characteristic == transfer
        assert decoder.metadata.color_space == "bt2020nc"
        assert decoder.metadata.color_primaries == "bt2020"
        batch = decoder.get_frames_at([2, 0, 2])
        assert batch.data.dtype == expected_dtype
        assert decoder.get_frames_at([]).data.shape == ((0, 24, 32, 3) if order == "NHWC" else (0, 3, 24, 32))
    expected = ffmpeg_rgb(path, "rgb24" if expected_dtype == np.uint8 else "rgb48le", (3, 24, 32, 3))
    if expected_dtype == np.float32:
        expected = expected.astype(np.float32) / 65535
    if order == "NCHW":
        expected = expected.transpose(0, 3, 1, 2)
    np.testing.assert_allclose(
        batch.data, expected[[2, 0, 2]], atol=1 / 65535 if expected_dtype == np.float32 else 1, rtol=0
    )
    np.testing.assert_allclose(batch.pts_seconds, [0.2, 0, 0.2])
    if depth > 8 and expected_dtype != np.uint8:
        assert np.unique(batch.data).size > 256


@pytest.mark.parametrize("angle", [90, 180, 270, -90])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("order", ["NHWC", "NCHW"])
def test_display_rotation(rotation_source, tmp_path, angle, dtype, order):
    path = tmp_path / "rotated.mp4"
    run_ffmpeg("-display_rotation", angle, "-i", rotation_source, "-c", "copy", path)
    assert probe_stream(path)["side_data_list"][0]["rotation"] % 360 == angle % 360
    with VideoDecoder(rotation_source, output_dtype=dtype, dimension_order="NHWC") as original:
        expected = np.rot90(original.get_frames_at([2, 0, 2]).data, angle // 90, axes=(1, 2))
    with VideoDecoder(path, output_dtype=dtype, dimension_order=order) as decoder:
        height, width = expected.shape[1:3]
        assert (decoder.metadata.width, decoder.metadata.height) == (width, height)
        assert decoder.metadata.rotation % 360 == angle % 360
        batch = decoder.get_frames_at([2, 0, 2])
        empty = decoder.get_frames_at([]).data
        if order == "NCHW":
            expected = expected.transpose(0, 3, 1, 2)
        assert empty.shape == (0, *expected.shape[1:])
        np.testing.assert_array_equal(batch.data, expected)
        assert all(stride >= 0 for stride in batch.data.strides)
        np.testing.assert_allclose(batch.pts_seconds, [0.2, 0, 0.2])
        np.testing.assert_array_equal(np.from_dlpack(batch.data), expected)
    if dtype == np.uint8 and order == "NHWC":
        reference = ffmpeg_rgb(path, "rgb24", (3, height, width, 3))
        # The CLI converts after rotating, so on aarch64 FFmpeg 8 only one side may take
        # swscale's NEON path (widths that are multiples of 16), which rounds differently.
        np.testing.assert_allclose(batch.data, reference[[2, 0, 2]], atol=2, rtol=0)


def test_non_right_angle_is_explicit(rotation_source, tmp_path):
    path = tmp_path / "diagonal.mp4"
    run_ffmpeg("-display_rotation", "45", "-i", rotation_source, "-c", "copy", path)
    with pytest.raises(NotImplementedError, match="multiples of 90"):
        VideoDecoder(path)


def test_reflected_matrix_is_explicit(rotation_source, tmp_path):
    path = tmp_path / "reflected.mp4"
    run_ffmpeg("-display_hflip", "-i", rotation_source, "-c", "copy", path)
    assert probe_stream(path)["side_data_list"][0]["side_data_type"] == "Display Matrix"
    with pytest.raises(NotImplementedError, match="reflected"):
        VideoDecoder(path)


@pytest.mark.parametrize("dtype", [np.float64, np.int16, "invalid", ">u2"])
def test_invalid_output_dtype(rotation_source, dtype):
    with pytest.raises(ValueError, match="output_dtype"):
        VideoDecoder(rotation_source, output_dtype=dtype)
