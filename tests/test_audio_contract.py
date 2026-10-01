import numpy as np
import pytest

from tests.utils import as_numpy


def test_all_samples(backend, audio):
    path, expected, rate = audio
    dec = backend.AudioDecoder(path)
    samples = dec.get_all_samples()
    np.testing.assert_allclose(as_numpy(samples.data), expected, atol=1e-7, rtol=0)
    assert samples.sample_rate == rate
    assert samples.pts_seconds == 0
    assert samples.duration_seconds == 1
    assert dec.metadata.num_channels == 2
    assert dec.metadata.sample_rate == rate


@pytest.mark.parametrize("start,stop", [(0, 0.125), (0.123, 0.321), (0.5, None), (0, None)])
def test_sample_ranges_and_repeated_requests(backend, audio, start, stop):
    path, expected, rate = audio
    dec = backend.AudioDecoder(path)
    for _ in range(2):
        samples = dec.get_samples_played_in_range(start, stop)
        begin = round(start * rate)
        end = expected.shape[1] if stop is None else round(stop * rate)
        np.testing.assert_allclose(as_numpy(samples.data), expected[:, begin:end], atol=1e-7, rtol=0)
        assert samples.pts_seconds == pytest.approx(start)
        assert samples.duration_seconds == pytest.approx((end - begin) / rate)


def test_invalid_audio_range(backend, audio):
    with pytest.raises(ValueError):
        backend.AudioDecoder(audio[0]).get_samples_played_in_range(0.5, 0.1)
