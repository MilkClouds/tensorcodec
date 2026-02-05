"""Tests for VideoDecoder."""

import numpy as np
import pytest

from avdec import Frame, FrameBatch, SeekMode, VideoDecoder, VideoStreamMetadata


class TestVideoDecoderBasic:
    """Basic functionality tests."""

    def test_open_close(self, sample_video):
        """Test basic open/close functionality."""
        decoder = VideoDecoder(sample_video)
        assert decoder.source == sample_video
        decoder.close()

    def test_context_manager(self, sample_video):
        """Test context manager protocol."""
        with VideoDecoder(sample_video) as decoder:
            assert decoder.source == sample_video

    def test_metadata(self, sample_video):
        """Test metadata extraction."""
        with VideoDecoder(sample_video) as decoder:
            meta = decoder.metadata
            assert isinstance(meta, VideoStreamMetadata)
            assert meta.width == 320
            assert meta.height == 240
            assert meta.average_fps == pytest.approx(30.0, rel=0.1)
            assert meta.num_frames == 30
            # TorchCodec extended fields
            assert meta.begin_stream_seconds == pytest.approx(0.0, abs=0.01)
            assert meta.end_stream_seconds is not None

    def test_len(self, sample_video):
        """Test __len__ returns frame count."""
        with VideoDecoder(sample_video) as decoder:
            assert len(decoder) == 30

    def test_getitem_single(self, sample_video):
        """Test single frame indexing (NCHW default)."""
        with VideoDecoder(sample_video) as decoder:
            frame = decoder[0]
            assert isinstance(frame, np.ndarray)
            # Default is NCHW: (C, H, W)
            assert frame.shape == (3, 240, 320)
            assert frame.dtype == np.uint8

    def test_getitem_single_nhwc(self, sample_video):
        """Test single frame indexing with NHWC."""
        with VideoDecoder(sample_video, dimension_order="NHWC") as decoder:
            frame = decoder[0]
            assert frame.shape == (240, 320, 3)

    def test_getitem_negative(self, sample_video):
        """Test negative indexing."""
        with VideoDecoder(sample_video) as decoder:
            frame = decoder[-1]
            # Default is NCHW: (C, H, W)
            assert frame.shape == (3, 240, 320)

    def test_get_frames_at(self, sample_video):
        """Test batch frame retrieval by index (NCHW default)."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_at([0, 10, 20])
            assert isinstance(batch, FrameBatch)
            assert len(batch) == 3
            # Default is NCHW: (N, C, H, W)
            assert batch.data.shape == (3, 3, 240, 320)
            assert len(batch.pts_seconds) == 3
            assert len(batch.duration_seconds) == 3

    def test_get_frames_at_nhwc(self, sample_video):
        """Test batch frame retrieval with NHWC."""
        with VideoDecoder(sample_video, dimension_order="NHWC") as decoder:
            batch = decoder.get_frames_at([0, 10, 20])
            assert batch.data.shape == (3, 240, 320, 3)

    def test_get_frames_at_empty(self, sample_video):
        """Test empty index list."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_at([])
            assert len(batch) == 0
            assert batch.data.shape[0] == 0

    def test_get_frame_at(self, sample_video):
        """Test single frame retrieval."""
        with VideoDecoder(sample_video) as decoder:
            frame = decoder.get_frame_at(0)
            assert isinstance(frame, Frame)
            assert frame.data.shape == (3, 240, 320)  # NCHW default
            assert frame.pts_seconds >= 0.0
            assert frame.duration_seconds > 0.0

    def test_get_frames_played_at(self, sample_video):
        """Test batch frame retrieval by timestamp."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.5])
            assert isinstance(batch, FrameBatch)
            assert len(batch) == 2

    def test_get_frame_played_at(self, sample_video):
        """Test single frame retrieval by timestamp."""
        with VideoDecoder(sample_video) as decoder:
            frame = decoder.get_frame_played_at(0.0)
            assert isinstance(frame, Frame)
            assert frame.data.shape == (3, 240, 320)  # NCHW default

    def test_get_frames_played_at_exceeds_duration(self, sample_video):
        """Test error when timestamp exceeds duration."""
        with VideoDecoder(sample_video) as decoder:
            # Now raises IndexError with TorchCodec-compatible message
            with pytest.raises(IndexError, match="Invalid pts in seconds"):
                decoder.get_frames_played_at([100.0])

    def test_get_frames_in_range(self, sample_video):
        """Test frame range retrieval."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_in_range(0, 5)
            assert len(batch) == 5
            assert batch.data.shape == (5, 3, 240, 320)  # NCHW default

    def test_get_frames_played_in_range(self, sample_video):
        """Test frame range retrieval by timestamp."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_in_range(0.0, 0.5)
            assert len(batch) > 0


class TestSeekMode:
    """Test different seek modes."""

    def test_exact_mode(self, sample_video):
        """Test exact seek mode builds frame index."""
        with VideoDecoder(sample_video, seek_mode=SeekMode.EXACT) as decoder:
            assert decoder._frame_index is not None
            assert len(decoder._frame_index) == 30

    def test_approximate_mode(self, sample_video):
        """Test approximate seek mode skips frame index."""
        with VideoDecoder(sample_video, seek_mode=SeekMode.APPROXIMATE) as decoder:
            assert decoder._frame_index is None
            # Should still work using FPS-based calculation
            frame = decoder[0]
            # Default is NCHW: (C, H, W)
            assert frame.shape == (3, 240, 320)


class TestDimensionOrder:
    """Test dimension order parameter."""

    def test_invalid_dimension_order(self, sample_video):
        """Test invalid dimension order raises error."""
        with pytest.raises(ValueError, match="Invalid dimension_order"):
            VideoDecoder(sample_video, dimension_order="INVALID")

