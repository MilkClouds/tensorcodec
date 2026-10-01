"""Value-preserving MediaRef layouts, checked against raw source samples."""

import io

import numpy as np
import pytest

from tensorcodec.decoders import VideoDecoder
from tests.utils import probe_stream, run_ffmpeg

FORMATS = {
    "gray": (1, 8, "u1"),
    "gray12le": (1, 12, "<u2"),
    "gray16le": (1, 16, "<u2"),
    "gray16be": (1, 16, ">u2"),
    "rgb24": (3, 8, "u1"),
    "rgba": (4, 8, "u1"),
}


@pytest.fixture(params=FORMATS)
def native_video(request, tmp_path):
    fmt = request.param
    channels, depth, storage = FORMATS[fmt]
    # Odd row widths exercise FFmpeg's padded linesizes. Include range endpoints.
    values = np.arange(4 * 11 * 19 * channels, dtype=np.uint32).reshape(4, 11, 19, channels)
    values = (values * 127 + np.arange(4)[:, None, None, None]) % (1 << depth)
    values[:, 0, :4, 0] = [0, 1, (1 << depth) - 2, (1 << depth) - 1]
    expected = values.astype(np.uint8 if depth == 8 else np.uint16)
    raw, path = tmp_path / "pixels.raw", tmp_path / "pixels.nut"
    raw.write_bytes(expected.astype(storage).tobytes())
    run_ffmpeg(
        "-f",
        "rawvideo",
        "-pixel_format",
        fmt,
        "-video_size",
        "19x11",
        "-framerate",
        10,
        "-i",
        raw,
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        fmt,
        "-threads",
        1,
        path,
    )
    assert probe_stream(path)["pix_fmt"] == fmt
    return path, fmt, expected


@pytest.mark.parametrize("order", ["NCHW", "NHWC"])
def test_native_playback_and_ownership(native_video, order):
    path, fmt, expected = native_video
    if order == "NCHW":
        expected = expected.transpose(0, 3, 1, 2)
    source = io.BytesIO(path.read_bytes())
    with VideoDecoder(source, output_format="native", expected_pixel_format=fmt, dimension_order=order) as decoder:
        assert decoder.metadata.pixel_format == fmt
        batch = decoder.get_frames_played_at([0.29, 0.0, 0.29, 0.39])
        assert batch.pixel_format == fmt and batch.data.dtype == expected.dtype
        np.testing.assert_array_equal(batch.data, expected[[2, 0, 2, 3]])
        np.testing.assert_allclose(batch.pts_seconds, [0.2, 0, 0.2, 0.3], atol=1e-12)
        np.testing.assert_allclose(batch.duration_seconds, 0.1, atol=1e-12)
        np.testing.assert_array_equal(decoder.get_all_frames().data, expected)
        np.testing.assert_array_equal(decoder[1:4:2], expected[[1, 3]])
        assert decoder.get_frame_at(1).pixel_format == fmt
        assert batch[:2].pixel_format == fmt
        empty = decoder.get_frames_at([])
        assert empty.data.shape == (0, *expected.shape[1:])
        assert empty.pixel_format == fmt and empty.data.dtype == expected.dtype
        selected = decoder.get_frames_played_in_range(0.15, 0.3)
        np.testing.assert_array_equal(selected.data, expected[[1, 2]])
        resampled = decoder.get_frames_played_in_range(0, 0.4, fps=20)
        assert resampled.pixel_format == fmt
        np.testing.assert_array_equal(resampled.data, np.repeat(expected, 2, axis=0))
    assert not source.closed
    batch.data[0] = 0
    np.testing.assert_array_equal(batch.data[2], expected[2])


def test_native_dtype_and_format_validation(native_video):
    path, fmt, expected = native_video
    for dtype in ("auto", expected.dtype):
        with VideoDecoder(path, output_format="native", output_dtype=dtype) as decoder:
            np.testing.assert_array_equal(decoder[0], expected[0].transpose(2, 0, 1))
    for dtype in (np.float32, np.uint8 if expected.dtype == np.uint16 else np.uint16):
        with pytest.raises(ValueError, match="requires output_dtype"):
            VideoDecoder(path, output_format="native", output_dtype=dtype)
    with pytest.raises(ValueError, match="Expected source"):
        VideoDecoder(path, output_format="native", expected_pixel_format="yuv420p")
    with pytest.raises(ValueError, match="requires native"):
        VideoDecoder(path, expected_pixel_format=fmt)
    with pytest.raises(ValueError, match="output_format"):
        VideoDecoder(path, output_format="unknown")


def test_unsupported_planar_native_format_is_explicit(videos):
    with pytest.raises(ValueError, match="does not support"):
        VideoDecoder(videos["cfr"].path, output_format="native")
    with VideoDecoder(videos["cfr"].path) as decoder:
        assert decoder[0].shape[0] == 3


def test_native_vfr_offset_file_and_array_sources(native_video, tmp_path):
    path, fmt, expected = native_video
    vfr = tmp_path / "vfr.nut"
    run_ffmpeg(
        "-i",
        path,
        "-vf",
        "setpts=if(lt(N\\,2)\\,N\\,2*N)/(10*TB)",
        "-fps_mode",
        "vfr",
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        fmt,
        "-threads",
        1,
        "-output_ts_offset",
        2,
        vfr,
    )
    for source in (vfr.as_uri(), vfr.read_bytes(), np.frombuffer(vfr.read_bytes(), dtype=np.uint8)):
        with VideoDecoder(source, output_format="native") as decoder:
            np.testing.assert_allclose(decoder.get_all_frames().pts_seconds, [2, 2.1, 2.4, 2.6], atol=1e-12)
            batch = decoder.get_frames_played_at([2.55, 2.05, 2.35, 2.55])
            np.testing.assert_array_equal(batch.data, expected[[2, 0, 1, 2]].transpose(0, 3, 1, 2))
            np.testing.assert_array_equal(
                decoder.get_frames_played_in_range(2.35, 2.6).data, expected[[1, 2]].transpose(0, 3, 1, 2)
            )


@pytest.mark.parametrize("matrix", ["rotation", "reflection"])
def test_native_preserves_pixel_coordinates(tmp_path, matrix):
    expected = np.arange(11 * 19 * 3, dtype=np.uint8).reshape(11, 19, 3)
    raw = tmp_path / "rgb.raw"
    raw.write_bytes(expected.tobytes())
    source, rotated = tmp_path / "source.mov", tmp_path / "rotated.mov"
    run_ffmpeg(
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "19x11",
        "-framerate",
        10,
        "-i",
        raw,
        "-c:v",
        "png",
        "-threads",
        1,
        source,
    )
    options = ["-display_rotation", 45] if matrix == "rotation" else ["-display_hflip"]
    run_ffmpeg(*options, "-i", source, "-c", "copy", rotated)
    with VideoDecoder(rotated, output_format="native") as decoder:
        assert (decoder.metadata.height, decoder.metadata.width) == (11, 19)
        np.testing.assert_array_equal(decoder[0], expected.transpose(2, 0, 1))
