"""Time-only playback without an initial packet scan."""

import io

import numpy as np
import pytest

from benchmarks.open_cost import measure
from tensorcodec.decoders import VideoDecoder
from tests.utils import as_numpy, run_ffmpeg


def test_timestamp_boundaries_order_and_repeated_calls(video, oracle):
    times = np.concatenate((video.pts, video.pts + video.durations / 2, np.nextafter(video.pts[1:], -np.inf)))
    times = times[::-1].tolist() + [float(video.pts[0])] * 2
    with VideoDecoder(video.path, stream_index=video.stream_index, seek_mode="timestamp") as decoder:
        reference = oracle.VideoDecoder(video.path, stream_index=video.stream_index)
        for seconds in (times, [float(video.pts[0])], times):
            batch = decoder.get_frames_played_at(seconds)
            expected = reference.get_frames_played_at(seconds)
            np.testing.assert_array_equal(batch.data, as_numpy(expected.data))
            np.testing.assert_array_equal(batch.pts_seconds, as_numpy(expected.pts_seconds))
            np.testing.assert_array_equal(batch.duration_seconds, as_numpy(expected.duration_seconds))
        assert decoder.metadata.num_frames is None
        for operation in (
            lambda: len(decoder),
            lambda: decoder[0],
            lambda: decoder[:],
            decoder.get_all_frames,
            lambda: decoder.get_frames_played_in_range(0, 1),
        ):
            with pytest.raises(NotImplementedError):
                operation()
        for seconds in ([float("nan")], [float("inf")]):
            with pytest.raises(ValueError):
                decoder.get_frames_played_at(seconds)
        with pytest.raises(RuntimeError):
            decoder.get_frames_played_at([float(video.pts[0] - 1)])
        with pytest.raises(RuntimeError):
            decoder.get_frames_played_at([float(video.end + 1)])
        empty = decoder.get_frames_played_at([])
        assert empty.data.shape == (0, 3, 24, 32)
        batch = decoder.get_frames_played_at([float(video.pts[0])] * 2)
        batch.data[0] = 0
        assert batch.data[1].any()
    with pytest.raises(RuntimeError, match="closed"):
        decoder.get_frames_played_at([])


def test_timestamp_native_and_fps(tmp_path):
    pixels = np.arange(5 * 11 * 19, dtype=np.uint16).reshape(5, 11, 19)
    raw, path = tmp_path / "gray.raw", tmp_path / "gray.nut"
    raw.write_bytes(pixels.astype(">u2").tobytes())
    run_ffmpeg(
        "-f",
        "rawvideo",
        "-pixel_format",
        "gray16be",
        "-video_size",
        "19x11",
        "-framerate",
        10,
        "-i",
        raw,
        "-c:v",
        "rawvideo",
        "-threads",
        1,
        path,
    )
    source = io.BytesIO(path.read_bytes())
    with VideoDecoder(source, seek_mode="timestamp", output_format="native", dimension_order="NHWC") as decoder:
        result = decoder.get_frames_played_at([0.35, 0.05, 0.35, 0.45])
        np.testing.assert_array_equal(result.data[..., 0], pixels[[3, 0, 3, 4]])
        assert result.pixel_format == "gray16be"
        result = decoder.get_frames_played_in_range(0, 0.5, fps=20)
        np.testing.assert_array_equal(result.data[..., 0], np.repeat(pixels, 2, axis=0))
        np.testing.assert_allclose(result.duration_seconds, 0.05)
    assert not source.closed


def test_timestamp_avoids_open_scan_and_repeated_window_reads(tmp_path):
    raw, path = tmp_path / "noise.raw", tmp_path / "noise.mkv"
    pixels = np.random.default_rng(1).integers(0, 256, (100, 64, 64, 3), dtype=np.uint8)
    raw.write_bytes(pixels.tobytes())
    run_ffmpeg(
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "64x64",
        "-framerate",
        10,
        "-i",
        raw,
        "-c:v",
        "ffv1",
        "-g",
        10,
        "-threads",
        1,
        path,
    )
    times = [8 + i / 10 for i in range(11)]
    exact, reference = measure(VideoDecoder, path, times)
    timestamp, actual = measure(VideoDecoder, path, times, seek_mode="timestamp")
    for a, b in zip(reference, actual):
        np.testing.assert_array_equal(a, b)
    assert timestamp["open_bytes"] < exact["open_bytes"] / 2
    assert timestamp["window_bytes"] < path.stat().st_size / 2


def test_timestamp_rejects_mappings(videos):
    with pytest.raises(ValueError, match="requires exact"):
        VideoDecoder(videos["cfr"].path, seek_mode="timestamp", custom_frame_mappings=b"{}")


def test_timestamp_backs_off_when_mkv_cue_points_past_target(tmp_path):
    import json
    import subprocess

    path = tmp_path / "bad-cues.mkv"
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x48:rate=10",
        "-t",
        4,
        "-c:v",
        "ffv1",
        "-g",
        10,
        "-threads",
        1,
        "-cluster_time_limit",
        1000,
        path,
    )
    data = bytearray(path.read_bytes())

    def vint(offset, keep_marker=False):
        width = 1
        while not data[offset] & (1 << (8 - width)):
            width += 1
        value = int.from_bytes(data[offset : offset + width], "big")
        return (value if keep_marker else value & ((1 << (7 * width)) - 1)), offset + width

    def shift_cues(start, end):
        changed = 0
        while start < end:
            tag, offset = vint(start, True)
            size, offset = vint(offset)
            if tag == 0xBB:
                changed += shift_cues(offset, offset + size)
            elif tag == 0xB3:
                value = int.from_bytes(data[offset : offset + size], "big")
                data[offset : offset + size] = (max(1, value - 1000) if value else 0).to_bytes(size, "big")
                changed += 1
            start = offset + size
        return changed

    # Keep the encoded frames intact, but make each later cue advertise an earlier time.
    position = data.rfind(bytes.fromhex("1c53bb6b"))
    size, start = vint(position + 4)
    assert shift_cues(start, start + size) == 4
    path.write_bytes(data)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-read_intervals",
            "1.5%+0.1",
            "-show_frames",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    assert float(json.loads(probe.stdout)["frames"][0]["pts_time"]) > 1.5
    with VideoDecoder(path, seek_mode="timestamp") as decoder:
        batch = decoder.get_frames_played_at([1.5, 2.5, 0.5, 1.5])
        np.testing.assert_allclose(batch.pts_seconds, [1.5, 2.5, 0.5, 1.5])


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize("order", ["NCHW", "NHWC"])
def test_timestamp_conversion_matches_exact(videos, dtype, order):
    options = {"output_dtype": dtype, "dimension_order": order}
    with VideoDecoder(videos["bframes"].path, **options) as exact:
        expected = exact.get_frames_played_at([0.95, 0.05, 0.95])
    with VideoDecoder(videos["bframes"].path, seek_mode="timestamp", **options) as decoder:
        np.testing.assert_array_equal(decoder.get_frames_played_at([0.95, 0.05, 0.95]).data, expected.data)
