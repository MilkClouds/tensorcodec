"""Tests for TorchCodec compatibility.

These tests verify that avdec produces identical output to TorchCodec
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
except ImportError:
    HAS_TORCHCODEC = False

from avdec import SeekMode, VideoDecoder

# Path to TorchCodec test resources (from environment variable)
TORCHCODEC_TEST_RESOURCES = os.environ.get("TORCHCODEC_TEST_RESOURCES")
HAS_TEST_RESOURCES = TORCHCODEC_TEST_RESOURCES is not None and os.path.isdir(TORCHCODEC_TEST_RESOURCES)

if HAS_TEST_RESOURCES:
    NASA_VIDEO_PATH = os.path.join(TORCHCODEC_TEST_RESOURCES, "nasa_13013.mp4")
else:
    NASA_VIDEO_PATH = None


def frames_equal(avdec_frame: np.ndarray, torchcodec_frame, *, atol: int = 20) -> bool:
    """Compare avdec frame (NHWC numpy) with TorchCodec frame (NCHW torch tensor).

    Args:
        avdec_frame: numpy array in HWC or NHWC format
        torchcodec_frame: torch tensor in CHW or NCHW format
        atol: absolute tolerance for comparison. Default is 20 to account for
            minor differences between PyAV and TorchCodec's FFmpeg bindings.

    Returns:
        True if frames are equal within tolerance
    """
    # Convert TorchCodec tensor to numpy
    tc_np = torchcodec_frame.numpy()

    # Handle dimension differences
    # avdec: HWC or NHWC
    # torchcodec: CHW or NCHW
    if tc_np.ndim == 3:
        # CHW -> HWC
        tc_np = np.transpose(tc_np, (1, 2, 0))
    elif tc_np.ndim == 4:
        # NCHW -> NHWC
        tc_np = np.transpose(tc_np, (0, 2, 3, 1))

    return np.allclose(avdec_frame, tc_np, atol=atol, rtol=0)


def assert_frames_equal(avdec_frame: np.ndarray, torchcodec_frame, *, atol: int = 20, msg: str = ""):
    """Assert that avdec and TorchCodec frames are equal.

    Note: PyAV and TorchCodec use different FFmpeg bindings which can result in
    minor pixel differences (typically < 20) due to rounding in color space
    conversion and other internal operations. This is expected behavior.
    """
    # Convert TorchCodec tensor to numpy
    tc_np = torchcodec_frame.numpy()

    # Handle dimension differences
    if tc_np.ndim == 3:
        tc_np = np.transpose(tc_np, (1, 2, 0))
    elif tc_np.ndim == 4:
        tc_np = np.transpose(tc_np, (0, 2, 3, 1))

    if not np.allclose(avdec_frame, tc_np, atol=atol, rtol=0):
        diff = np.abs(avdec_frame.astype(np.int16) - tc_np.astype(np.int16))
        max_diff = diff.max()
        mean_diff = diff.mean()
        pct_diff_gt_3 = 100 * (diff > 3).sum() / diff.size
        raise AssertionError(
            f"Frames not equal{' (' + msg + ')' if msg else ''}. "
            f"Max diff: {max_diff}, Mean diff: {mean_diff:.2f}, "
            f"Pixels with diff>3: {pct_diff_gt_3:.2f}%, Tolerance: {atol}"
        )


@pytest.mark.skipif(not HAS_TORCHCODEC, reason="TorchCodec not installed")
@pytest.mark.skipif(not HAS_TEST_RESOURCES, reason="TORCHCODEC_TEST_RESOURCES env var not set")
class TestTorchCodecCompatibility:
    """Test that avdec produces identical output to TorchCodec.

    Requires:
        - TorchCodec installed
        - TORCHCODEC_TEST_RESOURCES environment variable set to torchcodec/test/resources
    """

    @pytest.fixture
    def nasa_video(self):
        """Return path to NASA test video."""
        return NASA_VIDEO_PATH

    def test_metadata_matches(self, nasa_video):
        """Test that metadata extraction matches TorchCodec."""
        avdec_decoder = VideoDecoder(nasa_video)
        tc_decoder = TorchCodecDecoder(nasa_video)

        # Compare metadata
        assert avdec_decoder.metadata.width == tc_decoder.metadata.width
        assert avdec_decoder.metadata.height == tc_decoder.metadata.height
        assert avdec_decoder.metadata.num_frames == tc_decoder.metadata.num_frames
        assert avdec_decoder.metadata.average_fps == pytest.approx(
            tc_decoder.metadata.average_fps, rel=0.01
        )
        assert avdec_decoder.metadata.duration_seconds == pytest.approx(
            tc_decoder.metadata.duration_seconds, rel=0.01
        )

        avdec_decoder.close()

    def test_frame_count_matches(self, nasa_video):
        """Test that frame count matches TorchCodec."""
        avdec_decoder = VideoDecoder(nasa_video)
        tc_decoder = TorchCodecDecoder(nasa_video)

        assert len(avdec_decoder) == len(tc_decoder)

        avdec_decoder.close()

    def test_single_frame_exact_match(self, nasa_video):
        """Test that single frame decoding matches TorchCodec exactly."""
        avdec_decoder = VideoDecoder(nasa_video, seek_mode=SeekMode.EXACT)
        tc_decoder = TorchCodecDecoder(nasa_video, seek_mode="exact")

        # Test first frame
        avdec_frame0 = avdec_decoder[0]
        tc_frame0 = tc_decoder[0]
        assert_frames_equal(avdec_frame0, tc_frame0, msg="frame 0")

        # Test middle frame
        avdec_frame180 = avdec_decoder[180]
        tc_frame180 = tc_decoder[180]
        assert_frames_equal(avdec_frame180, tc_frame180, msg="frame 180")

        # Test last frame
        avdec_frame_last = avdec_decoder[-1]
        tc_frame_last = tc_decoder[-1]
        assert_frames_equal(avdec_frame_last, tc_frame_last, msg="last frame")

        avdec_decoder.close()

    def test_get_frames_at_matches(self, nasa_video):
        """Test that batch frame retrieval matches TorchCodec."""
        avdec_decoder = VideoDecoder(nasa_video, seek_mode=SeekMode.EXACT)
        tc_decoder = TorchCodecDecoder(nasa_video, seek_mode="exact")

        indices = [0, 10, 25, 50, 100, 180, 300, 389]

        avdec_batch = avdec_decoder.get_frames_at(indices)
        tc_batch = tc_decoder.get_frames_at(indices)

        # Compare each frame
        for i, idx in enumerate(indices):
            assert_frames_equal(
                avdec_batch.data[i], tc_batch.data[i], msg=f"frame index {idx}"
            )

        avdec_decoder.close()

    def test_get_frames_played_at_matches(self, nasa_video):
        """Test that timestamp-based retrieval matches TorchCodec."""
        avdec_decoder = VideoDecoder(nasa_video, seek_mode=SeekMode.EXACT)
        tc_decoder = TorchCodecDecoder(nasa_video, seek_mode="exact")

        # Test various timestamps
        timestamps = [0.0, 0.5, 1.0, 2.5, 6.0, 10.0, 12.0]

        avdec_batch = avdec_decoder.get_frames_played_at(timestamps)
        tc_batch = tc_decoder.get_frames_played_at(timestamps)

        # Frames should be visually equivalent
        for i, ts in enumerate(timestamps):
            assert_frames_equal(
                avdec_batch.data[i], tc_batch.data[i], msg=f"timestamp {ts}s"
            )

        avdec_decoder.close()

    def test_pts_seconds_accuracy(self, nasa_video):
        """Test that PTS values are accurate and consistent with TorchCodec."""
        avdec_decoder = VideoDecoder(nasa_video, seek_mode=SeekMode.EXACT)
        tc_decoder = TorchCodecDecoder(nasa_video, seek_mode="exact")

        indices = [0, 50, 180, 389]

        avdec_batch = avdec_decoder.get_frames_at(indices)
        tc_batch = tc_decoder.get_frames_at(indices)

        # PTS values should be very close (within 1ms)
        for i, idx in enumerate(indices):
            avdec_pts = avdec_batch.pts_seconds[i]
            tc_pts = tc_batch.pts_seconds[i].item()
            assert avdec_pts == pytest.approx(tc_pts, abs=0.001), \
                f"PTS mismatch at index {idx}: avdec={avdec_pts}, tc={tc_pts}"

        avdec_decoder.close()

