"""Tests for TorchCodec compatibility.

These tests verify that avdec produces output comparable to TorchCodec
for the same video inputs.

To run these tests, set the TORCHCODEC_TEST_RESOURCES environment variable
to the path of the TorchCodec test resources directory:

    export TORCHCODEC_TEST_RESOURCES=/path/to/torchcodec/test/resources
    pytest tests/test_torchcodec_compat.py
"""

import os

import numpy as np
import pytest

# Check if torchcodec is available
try:
    from torchcodec.decoders import VideoDecoder as TorchCodecDecoder

    HAS_TORCHCODEC = True
except (ImportError, RuntimeError):
    HAS_TORCHCODEC = False

from avdec import VideoDecoder

# Path to TorchCodec test resources (from environment variable)
TORCHCODEC_TEST_RESOURCES = os.environ.get("TORCHCODEC_TEST_RESOURCES")
HAS_TEST_RESOURCES = TORCHCODEC_TEST_RESOURCES is not None and os.path.isdir(
    TORCHCODEC_TEST_RESOURCES
)

if HAS_TEST_RESOURCES:
    NASA_VIDEO_PATH = os.path.join(TORCHCODEC_TEST_RESOURCES, "nasa_13013.mp4")
else:
    NASA_VIDEO_PATH = None


def assert_frames_close(
    avdec_nchw: np.ndarray,
    torchcodec_nchw,
    *,
    atol: int = 20,
    msg: str = "",
):
    """Assert avdec (NCHW numpy) ≈ TorchCodec (NCHW torch tensor).

    Both avdec and TorchCodec output NCHW tensors, so no transpose is needed.
    Tolerance is set to 20 to account for minor FFmpeg binding differences.
    """
    tc_np = torchcodec_nchw.numpy() if not isinstance(torchcodec_nchw, np.ndarray) else torchcodec_nchw

    if not np.allclose(avdec_nchw, tc_np, atol=atol, rtol=0):
        diff = np.abs(avdec_nchw.astype(np.int16) - tc_np.astype(np.int16))
        raise AssertionError(
            f"Frames not equal{' (' + msg + ')' if msg else ''}. "
            f"Max diff: {diff.max()}, Mean diff: {diff.mean():.2f}, "
            f"Pixels with diff>3: {100 * (diff > 3).sum() / diff.size:.2f}%, "
            f"Tolerance: {atol}"
        )


@pytest.mark.skipif(not HAS_TORCHCODEC, reason="TorchCodec not installed")
@pytest.mark.skipif(not HAS_TEST_RESOURCES, reason="TORCHCODEC_TEST_RESOURCES env var not set")
class TestTorchCodecCompatibility:
    """Test that avdec produces comparable output to TorchCodec.

    Requires:
        - TorchCodec installed
        - TORCHCODEC_TEST_RESOURCES environment variable set
    """

    @pytest.fixture
    def nasa_video(self):
        """Return path to NASA test video."""
        return NASA_VIDEO_PATH

    def test_metadata_matches(self, nasa_video):
        """Test that metadata extraction matches TorchCodec."""
        with VideoDecoder(nasa_video) as avdec_dec:
            tc_dec = TorchCodecDecoder(nasa_video)
            assert avdec_dec.metadata.width == tc_dec.metadata.width
            assert avdec_dec.metadata.height == tc_dec.metadata.height
            assert avdec_dec.metadata.num_frames == tc_dec.metadata.num_frames
            assert float(avdec_dec.metadata.average_rate) == pytest.approx(
                tc_dec.metadata.average_fps, rel=0.01
            )
            assert float(avdec_dec.metadata.duration_seconds) == pytest.approx(
                tc_dec.metadata.duration_seconds, rel=0.01
            )

    def test_get_frames_played_at_matches(self, nasa_video):
        """Test that timestamp-based retrieval matches TorchCodec."""
        timestamps = [0.0, 0.5, 1.0, 2.5, 6.0, 10.0, 12.0]

        with VideoDecoder(nasa_video) as avdec_dec:
            tc_dec = TorchCodecDecoder(nasa_video)

            avdec_batch = avdec_dec.get_frames_played_at(timestamps)
            tc_batch = tc_dec.get_frames_played_at(timestamps)

            for i, ts in enumerate(timestamps):
                assert_frames_close(
                    avdec_batch.data[i], tc_batch.data[i], msg=f"timestamp {ts}s"
                )

