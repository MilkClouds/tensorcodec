"""Tests for TorchCodec compatibility.

These tests verify that avdec produces output comparable to TorchCodec
for the same video inputs.

The test video (``tests/resources/nasa_13013.mp4``) is NASA public-domain
footage sourced from the torchcodec repository (BSD-3-Clause).

    pytest tests/test_torchcodec_compat.py
"""

import numpy as np
import pytest

# Check if torchcodec is available
try:
    from torchcodec.decoders import VideoDecoder as TorchCodecDecoder

    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False

from avdec import VideoDecoder


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
class TestTorchCodecCompatibility:
    """Test that avdec produces comparable output to TorchCodec.

    Requires TorchCodec to be installed.
    The ``nasa_video`` fixture is provided by conftest.py.
    """

    def test_metadata_matches(self, nasa_video):
        """Test that metadata extraction matches TorchCodec."""
        with VideoDecoder(nasa_video) as avdec_dec:
            # nasa_13013.mp4 has multiple video streams; avdec uses
            # streams.video[0] (stream index 0) so we tell torchcodec
            # to use the same stream for an apples-to-apples comparison.
            tc_dec = TorchCodecDecoder(nasa_video, stream_index=0)
            assert avdec_dec.metadata.width == tc_dec.metadata.width
            assert avdec_dec.metadata.height == tc_dec.metadata.height
            assert avdec_dec.metadata.num_frames == tc_dec.metadata.num_frames
            assert float(avdec_dec.metadata.average_rate) == pytest.approx(tc_dec.metadata.average_fps, rel=0.01)
            assert float(avdec_dec.metadata.duration_seconds) == pytest.approx(
                tc_dec.metadata.duration_seconds, rel=0.01
            )

    def test_get_frames_played_at_matches(self, nasa_video):
        """Test that timestamp-based retrieval matches TorchCodec."""
        timestamps = [0.0, 0.5, 1.0, 2.5, 6.0, 10.0, 12.0]

        with VideoDecoder(nasa_video) as avdec_dec:
            tc_dec = TorchCodecDecoder(nasa_video, stream_index=0)

            avdec_batch = avdec_dec.get_frames_played_at(timestamps)
            tc_batch = tc_dec.get_frames_played_at(timestamps)

            for i, ts in enumerate(timestamps):
                assert_frames_close(avdec_batch.data[i], tc_batch.data[i], msg=f"timestamp {ts}s")
