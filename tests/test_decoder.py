"""Tests for VideoDecoder.

Basic tests using a synthetic sample video created in conftest.py.
"""

import numpy as np
import pytest

from avdec import FrameBatch, VideoDecoder, VideoStreamMetadata


class TestVideoDecoderInit:
    """Test VideoDecoder initialization."""

    def test_open_video(self, sample_video):
        """Test that a video can be opened successfully."""
        decoder = VideoDecoder(sample_video)
        assert decoder.metadata.width == 320
        assert decoder.metadata.height == 240
        assert decoder.metadata.num_frames == 30
        assert float(decoder.metadata.average_rate) == pytest.approx(30.0, rel=0.1)
        decoder.close()

    def test_open_nonexistent_file(self, tmp_path):
        """Test that opening a nonexistent file raises an error."""
        with pytest.raises(Exception):
            VideoDecoder(tmp_path / "nonexistent.mp4")

    def test_context_manager(self, sample_video):
        """Test context manager protocol."""
        with VideoDecoder(sample_video) as decoder:
            assert decoder.metadata.num_frames == 30


class TestVideoDecoderMetadata:
    """Test metadata extraction."""

    def test_metadata_fields(self, sample_video):
        """Test that all metadata fields are populated."""
        with VideoDecoder(sample_video) as decoder:
            meta = decoder.metadata
            assert isinstance(meta, VideoStreamMetadata)
            assert meta.width == 320
            assert meta.height == 240
            assert meta.num_frames > 0
            assert float(meta.duration_seconds) > 0
            assert float(meta.average_rate) > 0
            assert float(meta.begin_stream_seconds) >= 0
            assert float(meta.end_stream_seconds) > float(meta.begin_stream_seconds)


class TestGetFramesPlayedAt:
    """Test get_frames_played_at method."""

    def test_single_timestamp(self, sample_video):
        """Test decoding a single frame by timestamp."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            assert isinstance(batch, FrameBatch)
            assert batch.data.shape == (1, 3, 240, 320)  # NCHW
            assert len(batch.pts_seconds) == 1
            assert len(batch.duration_seconds) == 1

    def test_multiple_timestamps(self, sample_video):
        """Test decoding multiple frames by timestamp."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_at([0.0, 0.1, 0.5])
            assert batch.data.shape[0] == 3
            assert batch.data.shape[1] == 3  # Channels
            assert len(batch.pts_seconds) == 3

    def test_empty_list(self, sample_video):
        """Test with empty timestamp list."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_at([])
            assert batch.data.shape[0] == 0

    def test_out_of_range_timestamp(self, sample_video):
        """Test that out-of-range timestamps raise ValueError."""
        with VideoDecoder(sample_video) as decoder:
            with pytest.raises(ValueError):
                decoder.get_frames_played_at([100.0])

    def test_negative_timestamp(self, sample_video):
        """Test that negative timestamps raise ValueError."""
        with VideoDecoder(sample_video) as decoder:
            with pytest.raises(ValueError):
                decoder.get_frames_played_at([-0.1])

    def test_playback_semantics(self, sample_video):
        """Test that playback semantics are correct.

        frame[i].pts <= timestamp < frame[i+1].pts should return frame[i].
        """
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            assert batch.pts_seconds[0] == pytest.approx(0.0, abs=0.01)

    def test_timestamp_just_before_end(self, sample_video):
        """Test timestamp just before end_stream_seconds succeeds."""
        with VideoDecoder(sample_video) as decoder:
            end = float(decoder.metadata.end_stream_seconds)
            batch = decoder.get_frames_played_at([end - 0.001])
            assert batch.data.shape[0] == 1

    def test_timestamp_at_end_raises(self, sample_video):
        """Test timestamp == end_stream_seconds raises ValueError."""
        with VideoDecoder(sample_video) as decoder:
            end = float(decoder.metadata.end_stream_seconds)
            with pytest.raises(ValueError):
                decoder.get_frames_played_at([end])


class TestGetFramesPlayedInRange:
    """Test get_frames_played_in_range method."""

    def test_basic_range(self, sample_video):
        """Test basic range decoding."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_in_range(0.0, 0.5)
            assert isinstance(batch, FrameBatch)
            assert batch.data.shape[0] > 0

    def test_full_range(self, sample_video):
        """Test decoding entire video."""
        with VideoDecoder(sample_video) as decoder:
            meta = decoder.metadata
            batch = decoder.get_frames_played_in_range(
                float(meta.begin_stream_seconds),
                float(meta.end_stream_seconds),
            )
            assert batch.data.shape[0] > 0

    def test_invalid_range(self, sample_video):
        """Test that start > stop raises ValueError."""
        with VideoDecoder(sample_video) as decoder:
            with pytest.raises(ValueError):
                decoder.get_frames_played_in_range(0.5, 0.1)

    def test_with_fps(self, sample_video):
        """Test resampled decoding with fps parameter."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_in_range(0.0, 0.5, fps=10.0)
            assert isinstance(batch, FrameBatch)
            # At 10fps over 0.5s we expect ~5 frames
            assert batch.data.shape[0] == 5


class TestFrameBatch:
    """Test FrameBatch properties."""

    def test_nchw_format(self, sample_video):
        """Test that output is NCHW."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            assert batch.data.shape == (1, 3, 240, 320)

    def test_dtype(self, sample_video):
        """Test that output dtype is uint8."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            assert batch.data.dtype == np.uint8

    def test_pts_dtype(self, sample_video):
        """Test that PTS is float64."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            assert batch.pts_seconds.dtype == np.float64

    def test_repr(self, sample_video):
        """Test FrameBatch __repr__."""
        with VideoDecoder(sample_video) as decoder:
            batch = decoder.get_frames_played_at([0.0])
            r = repr(batch)
            assert "FrameBatch" in r
            assert "data (shape)" in r
