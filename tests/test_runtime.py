"""TensorCodec-specific ownership and torch-free guarantees."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


def test_invalid_exact_mapping_does_not_return_a_different_frame(tensorcodec, videos):
    frames = [{"pts": 50, "duration": 100, "key_frame": 1}]
    with (
        tensorcodec.decoders.VideoDecoder(
            videos["cfr"].path, custom_frame_mappings=json.dumps({"frames": frames})
        ) as dec,
        pytest.raises(RuntimeError, match="exact PTS"),
    ):
        dec.get_frame_at(0)


@pytest.fixture
def tensorcodec(request):
    if request.config.getoption("--backend") != "tensorcodec":
        pytest.skip("Production-runtime tests apply to TensorCodec")
    import tensorcodec

    return tensorcodec


def test_arrays_survive_close_and_dlpack(tensorcodec, videos):
    dec = tensorcodec.decoders.VideoDecoder(videos["cfr"].path)
    batch = dec.get_frames_at([3, 0, 3])
    saved = batch.data.copy()
    dec.close()
    np.testing.assert_array_equal(batch.data, saved)
    shared = np.from_dlpack(batch.data)
    assert np.shares_memory(shared, batch.data)
    batch.data[0, 0, 0, 0] = 123
    assert shared[0, 0, 0, 0] == 123
    with pytest.raises(RuntimeError, match="closed"):
        dec.get_frame_at(0)
    dec.close()


def test_context_and_concurrent_instances(tensorcodec, videos):
    def decode(i):
        with tensorcodec.decoders.VideoDecoder(videos["bframes"].path) as dec:
            return dec.get_frame_at(i).pts_seconds

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(decode, range(12))) == pytest.approx(np.arange(12) / 10)


def test_import_and_decode_without_torch_or_pyav(tensorcodec, videos):
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchcodec', 'av'}:
            raise ImportError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from tensorcodec.decoders import VideoDecoder
with VideoDecoder(sys.argv[1]) as dec:
    assert dec.get_frame_at(0).data.shape == (3, 24, 32)
assert not {'torch', 'torchcodec', 'av'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, str(videos["cfr"].path)], check=True)


def test_unsupported_options_are_explicit(tensorcodec, videos):
    with pytest.raises(NotImplementedError):
        tensorcodec.decoders.VideoDecoder(videos["cfr"].path, device="cuda")
    with pytest.raises(ValueError, match="Unsupported transform"):
        tensorcodec.decoders.VideoDecoder(videos["cfr"].path, transforms=[object()])


def test_file_like_is_read_on_demand_and_remains_owned_by_caller(tensorcodec, videos):
    import io

    class Source(io.BytesIO):
        def read(self, size):
            assert size > 0, "do not eagerly read the entire source"
            return super().read(size)

    source = Source(videos["bframes"].path.read_bytes())
    with tensorcodec.decoders.VideoDecoder(source) as dec:
        assert dec.get_frame_at(11).pts_seconds == pytest.approx(1.1)
    assert not source.closed


def test_interleaved_access_is_deterministic(tensorcodec, videos):
    case = videos["vfr"]
    dec = tensorcodec.decoders.VideoDecoder(case.path)
    for i in [4, 1, 7, 0, 11, 5, 5]:
        t = float(case.pts[i] + 0.01)
        assert dec.get_frame_played_at(t).pts_seconds == pytest.approx(case.pts[i])
        assert dec.get_frames_played_at([t]).pts_seconds[0] == pytest.approx(case.pts[i])
        assert dec.get_frame_at(i).pts_seconds == pytest.approx(case.pts[i])


def test_file_like_reference_cycles_are_collectable(tensorcodec, videos):
    import gc
    import io
    import weakref

    source = io.BytesIO(videos["bframes"].path.read_bytes())
    dec = tensorcodec.decoders.VideoDecoder(source)
    source.decoder = dec
    reference = weakref.ref(dec)
    source_reference = weakref.ref(source)
    del dec, source
    gc.collect()
    assert reference() is None
    assert source_reference() is None
