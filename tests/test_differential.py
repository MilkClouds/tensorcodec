"""Oracle is required with --compare; import/version failures are fatal."""

import numpy as np
import pytest

from tensorcodec.decoders import VideoDecoder
from tests.utils import as_numpy, index_input, run_ffmpeg, time_input


def compare_batch(actual, expected):
    assert actual.data.shape == expected.data.shape
    np.testing.assert_allclose(actual.pts_seconds, as_numpy(expected.pts_seconds), atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual.duration_seconds, as_numpy(expected.duration_seconds), atol=1e-12, rtol=0)
    # Identical FFmpeg conversion settings should match; allow one rounding unit, not 20.
    np.testing.assert_allclose(actual.data, as_numpy(expected.data), atol=1, rtol=0)


def test_video_differential(backend, oracle, video):
    if backend.__name__.startswith("torchcodec"):
        pytest.skip("Self-comparison is not useful")
    actual = backend.VideoDecoder(video.path, stream_index=video.stream_index)
    expected = oracle.VideoDecoder(video.path, stream_index=video.stream_index)
    for field in [
        "num_frames",
        "begin_stream_seconds",
        "end_stream_seconds",
        "duration_seconds",
        "average_fps",
        "width",
        "height",
        "codec",
        "stream_index",
    ]:
        assert (
            getattr(actual.metadata, field) == pytest.approx(getattr(expected.metadata, field))
            if field != "codec"
            else actual.metadata.codec == expected.metadata.codec
        )
    for indices in [[11, 0, 5, 5, 2], [], list(range(12))]:
        compare_batch(actual.get_frames_at(indices), expected.get_frames_at(index_input(oracle, indices)))
    times = ((video.pts[:-1] + video.pts[1:]) / 2).tolist()[::-1]
    compare_batch(actual.get_frames_played_at(times), expected.get_frames_played_at(time_input(oracle, times)))
    compare_batch(
        actual.get_frames_played_in_range(float(video.pts[1]), float(video.pts[8])),
        expected.get_frames_played_in_range(float(video.pts[1]), float(video.pts[8])),
    )


@pytest.mark.parametrize("sample_rate,num_channels", [(None, None), (4000, 1), (16000, 2)])
def test_audio_differential(backend, oracle, audio, sample_rate, num_channels):
    if backend.__name__.startswith("torchcodec"):
        pytest.skip("Self-comparison is not useful")
    kwargs = {"sample_rate": sample_rate, "num_channels": num_channels}
    actual = backend.AudioDecoder(audio[0], **kwargs).get_samples_played_in_range(0.123, 0.789)
    expected = oracle.AudioDecoder(audio[0], **kwargs).get_samples_played_in_range(0.123, 0.789)
    np.testing.assert_allclose(actual.data, as_numpy(expected.data), atol=1e-6, rtol=0)
    assert actual.pts_seconds == expected.pts_seconds
    assert actual.duration_seconds == expected.duration_seconds
    assert actual.sample_rate == expected.sample_rate


@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_color_conversion_differential(backend, oracle, color_video, dtype):
    import torch

    if backend.__name__.startswith("torchcodec"):
        pytest.skip("Self-comparison is not useful")
    actual = backend.VideoDecoder(color_video, output_dtype=np.dtype(dtype))
    expected = oracle.VideoDecoder(color_video, output_dtype=getattr(torch, dtype))
    # Exact pixel conversion when both backends link the same FFmpeg version.
    for indices in ([11, 3, 3, 0], [4, 1, 9], [0, 11]):
        left, right = actual.get_frames_at(indices), expected.get_frames_at(indices)
        np.testing.assert_allclose(left.data, as_numpy(right.data), atol=1 if dtype == "uint8" else 1 / 65535, rtol=0)
        np.testing.assert_allclose(left.pts_seconds, as_numpy(right.pts_seconds), atol=1e-12, rtol=0)
    for field in ("color_space", "color_primaries", "color_transfer_characteristic", "pixel_format"):
        assert getattr(actual.metadata, field) == getattr(expected.metadata, field)


def test_compressed_audio_differential(backend, oracle, compressed_audio):
    if backend.__name__.startswith("torchcodec"):
        pytest.skip("Self-comparison is not useful")
    actual = backend.AudioDecoder(compressed_audio)
    expected = oracle.AudioDecoder(compressed_audio)
    for start, stop in [(0.0, None), (0.123, 0.789), (0.8, 1.0), (0.05, 0.1)]:
        if compressed_audio.suffix == ".mp3" and stop == 0.1:
            # MP3's first presentation time is .138125, after this entire range.
            with pytest.raises(RuntimeError):
                expected.get_samples_played_in_range(start, stop)
            with pytest.raises(RuntimeError):
                actual.get_samples_played_in_range(start, stop)
            continue
        left = actual.get_samples_played_in_range(start, stop)
        right = expected.get_samples_played_in_range(start, stop)
        assert left.data.shape == right.data.shape
        np.testing.assert_allclose(left.data, as_numpy(right.data), atol=1e-6, rtol=0)
        assert left.pts_seconds == pytest.approx(right.pts_seconds, abs=1e-12)
        assert left.duration_seconds == right.duration_seconds


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
