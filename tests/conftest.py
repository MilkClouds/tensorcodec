"""Pytest configuration and shared fixtures for avdec tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_video(tmp_path):
    """Create a sample video for testing using PyAV.

    Creates a 30-frame video at 30fps (1 second duration) with 320x240 resolution.
    Each frame has a different grayscale value for easy verification.
    """
    import av

    video_path = tmp_path / "test_video.mp4"

    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("h264", rate=30)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"

    for i in range(30):
        # Create a frame with different colors for each frame
        frame = av.VideoFrame.from_ndarray(
            np.full((240, 320, 3), fill_value=i * 8, dtype=np.uint8),
            format="rgb24",
        )
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return video_path
