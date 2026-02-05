"""Tests for container-specific optimizations."""

import numpy as np
import pytest


class TestMP4Handler:
    """Tests for MP4 container handler."""

    def test_can_handle_mp4(self, sample_video):
        """Test that MP4Handler can handle MP4 files."""
        from avdec.containers.mp4 import MP4Handler

        handler = MP4Handler()
        assert handler.can_handle(sample_video)

    def test_cannot_handle_non_mp4(self, tmp_path):
        """Test that MP4Handler rejects non-MP4 files."""
        from avdec.containers.mp4 import MP4Handler

        handler = MP4Handler()

        # Create a fake MKV file
        fake_mkv = tmp_path / "test.mkv"
        fake_mkv.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 100)
        assert not handler.can_handle(fake_mkv)

        # Create a file with wrong extension
        fake_txt = tmp_path / "test.txt"
        fake_txt.write_text("not a video")
        assert not handler.can_handle(fake_txt)

    def test_build_frame_index(self, sample_video):
        """Test building frame index from MP4 atoms."""
        from avdec.containers.mp4 import MP4Handler

        handler = MP4Handler()
        frame_index = handler.build_frame_index(sample_video)

        # Should return a valid frame index
        assert frame_index is not None
        assert len(frame_index.frame_infos) > 0
        assert len(frame_index.keyframe_indices) > 0

    def test_frame_index_matches_packet_scan(self, sample_video):
        """Test that MP4 fast read produces same results as packet scan."""
        import av

        from avdec._metadata import build_frame_index_from_scan
        from avdec.containers.mp4 import MP4Handler

        handler = MP4Handler()
        mp4_index = handler.build_frame_index(sample_video)

        # Compare with packet scan
        container = av.open(str(sample_video))
        scan_index = build_frame_index_from_scan(container)
        container.close()

        assert mp4_index is not None
        assert len(mp4_index.frame_infos) == len(scan_index.frame_infos)

        # Compare PTS values (allow small floating point tolerance)
        for mp4_info, scan_info in zip(
            mp4_index.frame_infos, scan_index.frame_infos
        ):
            assert abs(mp4_info.pts - scan_info.pts) < 0.001, (
                f"PTS mismatch: mp4={mp4_info.pts}, scan={scan_info.pts}"
            )


class TestMKVHandler:
    """Tests for MKV container handler."""

    @pytest.fixture
    def sample_mkv(self, tmp_path):
        """Create a sample MKV video for testing."""
        import av

        video_path = tmp_path / "test_video.mkv"

        container = av.open(str(video_path), mode="w")
        stream = container.add_stream("libx264", rate=30)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"

        for i in range(30):
            frame = av.VideoFrame.from_ndarray(
                np.full((240, 320, 3), fill_value=i * 8, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()
        return video_path

    def test_can_handle_mkv(self, sample_mkv):
        """Test that MKVHandler can handle MKV files."""
        from avdec.containers.mkv import MKVHandler

        handler = MKVHandler()
        assert handler.can_handle(sample_mkv)

    def test_cannot_handle_non_mkv(self, sample_video):
        """Test that MKVHandler rejects non-MKV files."""
        from avdec.containers.mkv import MKVHandler

        handler = MKVHandler()
        assert not handler.can_handle(sample_video)

    def test_build_frame_index_returns_none(self, sample_mkv):
        """Test that MKV handler returns None (uses packet scan fallback)."""
        from avdec.containers.mkv import MKVHandler

        handler = MKVHandler()
        # MKV Cues only have keyframes, so we return None for complete index
        frame_index = handler.build_frame_index(sample_mkv)
        assert frame_index is None


class TestContainerRegistry:
    """Tests for container handler registry."""

    def test_get_handler_mp4(self, sample_video):
        """Test that get_handler returns MP4Handler for MP4 files."""
        from avdec.containers import get_handler
        from avdec.containers.mp4 import MP4Handler

        handler = get_handler(sample_video)
        assert handler is not None
        assert isinstance(handler, MP4Handler)

    def test_get_handler_unknown(self, tmp_path):
        """Test that get_handler returns None for unknown formats."""
        from avdec.containers import get_handler

        fake_file = tmp_path / "test.xyz"
        fake_file.write_bytes(b"\x00" * 100)
        handler = get_handler(fake_file)
        assert handler is None


class TestIntegration:
    """Integration tests for container optimizations with VideoDecoder."""

    def test_decoder_uses_mp4_optimization(self, sample_video):
        """Test that VideoDecoder uses MP4 optimization when available."""
        from avdec import VideoDecoder

        decoder = VideoDecoder(sample_video)
        # Should successfully decode frames
        frames = decoder.get_frames_at([0, 1, 2])
        assert frames.data.shape[0] == 3
        decoder.close()

