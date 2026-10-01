"""Run these same expectations first against TorchCodec, then TensorCodec."""

import io

import numpy as np
import pytest

from tests.utils import as_numpy, index_input, time_input


def open_video(backend, case, **kwargs):
    return backend.VideoDecoder(case.path, stream_index=case.stream_index, **kwargs)


def assert_batch(batch, case, indices, order="NCHW"):
    indices = np.asarray(indices, dtype=np.int64)
    data = as_numpy(batch.data)
    expected_shape = (len(indices), 3, 24, 32) if order == "NCHW" else (len(indices), 24, 32, 3)
    assert data.shape == expected_shape
    assert data.dtype == np.uint8
    np.testing.assert_allclose(as_numpy(batch.pts_seconds), case.pts[indices], atol=1e-12, rtol=0)
    np.testing.assert_allclose(as_numpy(batch.duration_seconds), case.durations[indices], atol=1e-12, rtol=0)
    # Gray identities survive the lossless H.264 YUV/RGB conversion with <= 2 rounding units.
    for frame, index in zip(data, indices):
        np.testing.assert_allclose(frame, int(case.levels[index]), atol=2, rtol=0)


def test_exact_metadata(backend, video):
    dec = open_video(backend, video)
    assert len(dec) == len(video.pts)
    assert dec.stream_index == video.stream_index
    m = dec.metadata
    assert (m.width, m.height) == (32, 24)
    assert m.num_frames == len(video.pts)
    assert m.begin_stream_seconds == pytest.approx(video.pts[0], abs=1e-12)
    assert m.end_stream_seconds == pytest.approx(video.end, abs=1e-12)
    assert m.duration_seconds == pytest.approx(video.end - video.pts[0], abs=1e-12)
    assert m.average_fps == pytest.approx(len(video.pts) / m.duration_seconds)


@pytest.mark.parametrize("order", ["NCHW", "NHWC"])
def test_batch_indices_order_duplicates_empty(backend, video, order):
    dec = open_video(backend, video, dimension_order=order)
    for indices in ([7, 1, 7, 0, 11], [], [11], list(range(12))):
        assert_batch(dec.get_frames_at(index_input(backend, indices)), video, indices, order)


def test_playback_boundaries(backend, video):
    dec = open_video(backend, video)
    queries, expected = [], []
    for i, pts in enumerate(video.pts):
        next_pts = video.pts[i + 1] if i + 1 < len(video.pts) else video.end
        queries.extend([pts, pts + (next_pts - pts) / 2, np.nextafter(next_pts, -np.inf)])
        expected.extend([i, i, i])
    assert_batch(
        dec.get_frames_played_at(time_input(backend, queries[::-1] + queries[:3])),
        video,
        expected[::-1] + expected[:3],
    )
    assert_batch(dec.get_frames_played_at(time_input(backend, [])), video, [])


def test_single_frame_and_unpack(backend, video):
    dec = open_video(backend, video)
    for i in [0, 5, 11]:
        frame = dec.get_frame_at(i)
        data, pts, duration = frame
        assert as_numpy(data).shape == (3, 24, 32)
        assert pts == pytest.approx(video.pts[i], abs=1e-12)
        assert duration == pytest.approx(video.durations[i], abs=1e-12)
        played = dec.get_frame_played_at(float(video.pts[i]))
        np.testing.assert_array_equal(as_numpy(played.data), as_numpy(data))


def test_indexing_and_ranges(backend, video):
    dec = open_video(backend, video)
    np.testing.assert_array_equal(as_numpy(dec[-1]), as_numpy(dec.get_frame_at(11).data))
    np.testing.assert_array_equal(as_numpy(dec[np.int64(2)]), as_numpy(dec.get_frame_at(2).data))
    np.testing.assert_array_equal(as_numpy(dec[1:9:2]), as_numpy(dec.get_frames_at([1, 3, 5, 7]).data))
    for args, expected in [((1, 9, 2), [1, 3, 5, 7]), ((-4, 100, 1), [8, 9, 10, 11]), ((5, 5, 1), [])]:
        assert_batch(dec.get_frames_in_range(*args), video, expected)


def test_time_range_includes_frame_playing_at_start(backend, video):
    dec = open_video(backend, video)
    start = float((video.pts[1] + video.pts[2]) / 2)
    stop = float(video.pts[7])
    assert_batch(dec.get_frames_played_in_range(start, stop), video, [1, 2, 3, 4, 5, 6])
    assert_batch(dec.get_frames_played_in_range(start, start), video, [])
    assert_batch(dec.get_frames_played_in_range(float(video.pts[0]), video.end), video, range(12))


def test_batch_object_indexing(backend, video):
    batch = open_video(backend, video).get_frames_at([0, 2, 4])
    assert len(batch) == 3
    assert len(list(batch)) == 3
    assert as_numpy(batch[1].data).shape == (3, 24, 32)
    assert as_numpy(batch[1].pts_seconds).shape == ()
    assert len(batch[:2]) == 2
    assert "data (shape)" in repr(batch)


@pytest.mark.parametrize("kind", ["bytes", "file", "array"])
def test_encoded_sources(backend, videos, kind):
    case = videos["bframes"]
    encoded = case.path.read_bytes()
    source = encoded if kind == "bytes" else io.BytesIO(encoded)
    if kind == "array":
        if backend.__name__.startswith("torchcodec"):
            import torch

            source = torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8).copy())
        else:
            source = np.frombuffer(encoded, dtype=np.uint8)
    assert_batch(backend.VideoDecoder(source).get_frames_at([11, 0, 4]), case, [11, 0, 4])


@pytest.mark.parametrize("kwargs", [{"dimension_order": "CHWN"}, {"seek_mode": "wrong"}, {"num_ffmpeg_threads": None}])
def test_invalid_constructor_arguments(backend, videos, kwargs):
    with pytest.raises(ValueError):
        backend.VideoDecoder(videos["cfr"].path, **kwargs)


def test_absolute_stream_indices(backend, videos):
    case = videos["multistream"]
    assert_batch(backend.VideoDecoder(case.path).get_frames_at([0]), case, [0])
    for index in [0, 2]:
        with pytest.raises(ValueError):
            backend.VideoDecoder(case.path, stream_index=index)


def test_query_validation(backend, videos):
    case = videos["cfr"]
    dec = open_video(backend, case)
    for t in [-0.1, case.end, np.nan, np.inf]:
        with pytest.raises(IndexError):
            dec.get_frame_played_at(t)
    for start, stop in [(-0.1, 0.5), (0.8, 0.2), (0, 10), (case.end, case.end)]:
        with pytest.raises(ValueError):
            dec.get_frames_played_in_range(start, stop)
    with pytest.raises(TypeError):
        dec["bad"]
    with pytest.raises(ValueError):
        dec.get_frames_in_range(0, 3, 0)


def test_approximate_mode_cfr(backend, videos):
    case = videos["bframes"]
    dec = open_video(backend, case, seek_mode="approximate")
    assert_batch(dec.get_frames_at([0, 5, 11]), case, [0, 5, 11])
    assert_batch(dec.get_frames_played_at([0.05, 0.55, 1.15]), case, [0, 5, 11])


@pytest.mark.parametrize("fps", [5, 20])
def test_resampling_has_output_grid_timing(backend, videos, fps):
    case = videos["cfr"]
    dec = open_video(backend, case)
    batch = dec.get_frames_played_in_range(0.15, 0.65, fps=fps)
    times = 0.15 + np.arange(int(np.ceil((0.65 - 0.15) * fps))) / fps
    indices = np.searchsorted(case.pts, times, side="right") - 1
    np.testing.assert_allclose(as_numpy(batch.pts_seconds), times, atol=1e-12, rtol=0)
    np.testing.assert_allclose(as_numpy(batch.duration_seconds), 1 / fps, atol=1e-12, rtol=0)
    for frame, index in zip(as_numpy(batch.data), indices):
        np.testing.assert_allclose(frame, case.levels[index], atol=2, rtol=0)


def test_get_all_frames(backend, video):
    assert_batch(open_video(backend, video).get_all_frames(), video, range(12))


@pytest.mark.parametrize("dtype", ["uint8", "float32", "auto"])
def test_output_dtype(backend, videos, dtype):
    if backend.__name__.startswith("torchcodec"):
        import torch

        argument = dtype if dtype == "auto" else getattr(torch, dtype)
    else:
        argument = dtype if dtype == "auto" else np.dtype(dtype)
    dec = open_video(backend, videos["cfr"], output_dtype=argument)
    frame = as_numpy(dec.get_frame_at(0).data)
    assert frame.dtype == (np.float32 if dtype == "float32" else np.uint8)
    # Float output is converted at higher precision, not simply uint8 / 255.
    np.testing.assert_allclose(
        frame, 16 / 255 if dtype == "float32" else 16, atol=1 / 255 if dtype == "float32" else 0
    )


def test_custom_mappings(backend, videos):
    import json

    case = videos["cfr"]
    frames = [{"pts": i * 100, "duration": 100, "key_frame": int(i == 0)} for i in range(12)]
    dec = open_video(backend, case, custom_frame_mappings=json.dumps({"frames": frames}))
    assert_batch(dec.get_frames_at([11, 3, 0]), case, [11, 3, 0])


def test_index_bounds_and_negative_steps(backend, videos):
    dec = open_video(backend, videos["cfr"])
    for i in [-13, 12]:
        with pytest.raises(IndexError):
            dec.get_frame_at(i)
    with pytest.raises(RuntimeError):
        dec[::-1]
