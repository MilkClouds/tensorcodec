"""Precision and orientation checks against independent FFmpeg output."""

import json
import subprocess

import numpy as np
import pytest

from tensorcodec.decoders import VideoDecoder
from tests.conftest import run_ffmpeg


@pytest.fixture(
    scope="module",
    params=[
        (8, "bt709", "tv"),
        (10, "bt709", "tv"),
        (10, "smpte2084", "tv"),
        (12, "arib-std-b67", "tv"),
        (16, "bt709", "tv"),
        (8, "bt709", "pc"),
        (10, "smpte2084", "pc"),
    ],
)
def precision_video(request, tmp_path_factory):
    depth, transfer, color_range = request.param
    root = tmp_path_factory.mktemp("precision")
    raw = root / "input.raw"
    dtype = np.uint8 if depth == 8 else np.dtype("<u2")
    low, high = (0, (1 << depth) - 1) if color_range == "pc" else (16 << (depth - 8), 235 << (depth - 8))
    y = np.linspace(low, high, 32 * 24).reshape(24, 32).astype(dtype)
    chroma_u = np.broadcast_to(np.linspace(3 << (depth - 3), 5 << (depth - 3), 32).astype(dtype), y.shape)
    chroma_v = np.broadcast_to(np.linspace(5 << (depth - 3), 3 << (depth - 3), 24).astype(dtype)[:, None], y.shape)
    raw.write_bytes(np.stack([y, chroma_u, chroma_v]).tobytes() * 3)
    pixel_format = "yuv444p" if depth == 8 else f"yuv444p{depth}le"
    path = root / "video.mkv"
    run_ffmpeg(
        "-f",
        "rawvideo",
        "-pixel_format",
        pixel_format,
        "-video_size",
        "32x24",
        "-framerate",
        "10",
        "-i",
        raw,
        "-vf",
        f"setparams=color_primaries=bt2020:color_trc={transfer}:colorspace=bt2020nc",
        "-c:v",
        "ffv1",
        "-threads",
        "1",
        "-colorspace",
        "bt2020nc",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        transfer,
        "-color_range",
        color_range,
        path,
    )
    stream = probe_stream(path)
    assert stream["pix_fmt"] == pixel_format
    assert stream["color_transfer"] == transfer
    assert stream["color_primaries"] == "bt2020"
    assert stream["color_range"] == color_range
    return path, depth, transfer, color_range


def probe_stream(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def ffmpeg_rgb(path, pixel_format, shape):
    output = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-threads",
            "1",
            "-sws_flags",
            "0",
            "-pix_fmt",
            pixel_format,
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(output, dtype=np.uint8 if pixel_format == "rgb24" else "<u2").reshape(shape)


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


@pytest.fixture(scope="module")
def rotation_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("rotation")
    path = root / "source.mp4"
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=32x24:rate=10:duration=0.3",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-crf",
        "0",
        path,
    )
    return path


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
        np.testing.assert_allclose(batch.data, reference[[2, 0, 2]], atol=1, rtol=0)


def test_non_right_angle_is_explicit(rotation_source, tmp_path):
    path = tmp_path / "diagonal.mp4"
    run_ffmpeg("-display_rotation", "45", "-i", rotation_source, "-c", "copy", path)
    with pytest.raises(NotImplementedError, match="multiples of 90"):
        VideoDecoder(path)


@pytest.mark.parametrize("dtype", ["uint8", "float32", "auto"])
def test_precision_matches_pinned_oracle(precision_video, oracle, dtype):
    import torch

    path, _, _, _ = precision_video
    with VideoDecoder(path, output_dtype=dtype) as actual:
        expected = oracle.VideoDecoder(path, output_dtype=dtype if dtype == "auto" else getattr(torch, dtype))
        left = actual.get_frames_at([2, 0, 2])
        right = expected.get_frames_at([2, 0, 2])
        assert left.data.dtype == right.data.numpy().dtype
        tolerance = 1 if left.data.dtype == np.uint8 else 1 / 65535
        np.testing.assert_allclose(left.data, right.data.numpy(), atol=tolerance, rtol=0)
        np.testing.assert_allclose(left.pts_seconds, right.pts_seconds.numpy(), atol=1e-12, rtol=0)


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_rotation_matches_pinned_oracle(rotation_source, tmp_path, oracle, angle):
    path = tmp_path / "rotated.mp4"
    run_ffmpeg("-display_rotation", angle, "-i", rotation_source, "-c", "copy", path)
    with VideoDecoder(path) as actual:
        expected = oracle.VideoDecoder(path)
        assert actual.metadata.rotation == expected.metadata.rotation
        assert (actual.metadata.width, actual.metadata.height) == (expected.metadata.width, expected.metadata.height)
        np.testing.assert_allclose(actual[:], expected[:].numpy(), atol=1, rtol=0)


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
