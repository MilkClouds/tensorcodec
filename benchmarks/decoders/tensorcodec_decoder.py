"""Benchmark adapter for tensorcodec."""

from __future__ import annotations

import numpy as np

from tensorcodec.decoders import VideoDecoder


class TensorCodecDecoder:
    """Wraps :class:`tensorcodec.decoders.VideoDecoder` for the benchmark protocol."""

    name = "tensorcodec"

    def get_frames_played_at(self, video_path: str, seconds: list[float]) -> np.ndarray:
        with VideoDecoder(video_path) as dec:
            batch = dec.get_frames_played_at(seconds)
            return batch.data  # NCHW uint8

    def get_frames_played_in_range(
        self,
        video_path: str,
        start_seconds: float,
        stop_seconds: float,
        fps: float | None = None,
    ) -> np.ndarray:
        with VideoDecoder(video_path) as dec:
            batch = dec.get_frames_played_in_range(start_seconds, stop_seconds, fps=fps)
            return batch.data  # NCHW uint8

    def get_video_duration(self, video_path: str) -> float:
        with VideoDecoder(video_path) as dec:
            return float(dec.metadata.end_stream_seconds)
