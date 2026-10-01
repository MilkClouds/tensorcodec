"""Read accounting and reusable frame mappings across playback edge cases."""

import io
import json
import subprocess

import numpy as np

from benchmarks.open_cost import CountingFile, measure
from tensorcodec.decoders import VideoDecoder
from tests.utils import as_numpy, run_ffmpeg


def frame_mappings(path, stream_index=0):
    return subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            str(stream_index),
            "-show_frames",
            "-show_entries",
            "frame=pts,duration,key_frame",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
    ).stdout


def test_read_accounting_includes_rereads_and_short_reads():
    source = CountingFile(io.BytesIO(b"abcdefgh"))
    assert source.read(3) == b"abc"
    source.seek(-2, 1)
    assert source.read(20) == b"bcdefgh"
    assert source.read(1) == b""
    assert source.tell() == 8
    assert source.bytes_read == 10 and source.read_calls == 3


def test_precomputed_mappings_match_exact_and_oracle(video, oracle):
    mappings = frame_mappings(video.path, video.stream_index)
    assert len(json.loads(mappings)["frames"]) == len(video.pts)
    seconds = (video.pts + video.durations / 2)[[len(video.pts) - 1, 0, 3, 0]].tolist()
    options = {"stream_index": video.stream_index}
    _, expected = measure(VideoDecoder, video.path, seconds, **options)
    for cls in (VideoDecoder, oracle.VideoDecoder):
        _, actual = measure(cls, video.path, seconds, custom_frame_mappings=mappings, **options)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        decoder = cls(video.path, custom_frame_mappings=mappings, **options)
        try:
            # Explicit indices avoid TorchCodec 0.17's mapped offset end-time bug.
            frames = decoder.get_frames_at(list(range(len(video.pts))))
            np.testing.assert_allclose(as_numpy(frames.pts_seconds), video.pts, atol=1e-12)
            np.testing.assert_allclose(as_numpy(frames.duration_seconds), video.durations, atol=1e-12)
        finally:
            if hasattr(decoder, "close"):
                decoder.close()


def test_mappings_avoid_full_file_read_on_open(tmp_path):
    raw, path = tmp_path / "noise.raw", tmp_path / "noise.mkv"
    pixels = np.random.default_rng(0).integers(0, 256, (100, 128, 128, 3), dtype=np.uint8)
    raw.write_bytes(pixels.tobytes())
    run_ffmpeg(
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "128x128",
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
    exact, expected = measure(VideoDecoder, path, [9.5, 0.05, 9.5])
    mapped, actual = measure(VideoDecoder, path, [9.5, 0.05, 9.5], custom_frame_mappings=frame_mappings(path))
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    assert exact["open_bytes"] >= path.stat().st_size * 0.9
    assert mapped["open_bytes"] < path.stat().st_size * 0.5
