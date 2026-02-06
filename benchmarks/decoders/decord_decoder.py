"""Benchmark adapter for Decord.

Requires ``decord`` to be installed.  The benchmark registry in
``__init__.py`` catches the ``ImportError`` if it is missing.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from decord import VideoReader, cpu


class DecordDecoder:
    """Wraps :class:`decord.VideoReader` for the benchmark protocol."""

    name = "decord"

    def get_frames_played_at(
        self, video_path: str, seconds: List[float]
    ) -> np.ndarray:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        # Convert timestamps to frame indices using playback-frame semantics:
        # frame[i].pts <= timestamp < frame[i+1].pts
        indices = [min(int(t * fps), len(vr) - 1) for t in seconds]
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
        return np.transpose(frames, (0, 3, 1, 2))  # -> NCHW

    def get_frames_played_in_range(
        self,
        video_path: str,
        start_seconds: float,
        stop_seconds: float,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        vr = VideoReader(video_path, ctx=cpu(0))
        avg_fps = vr.get_avg_fps()
        start_idx = int(start_seconds * avg_fps)
        stop_idx = min(int(stop_seconds * avg_fps), len(vr))
        indices = list(range(start_idx, stop_idx))
        if not indices:
            return np.empty((0, 3, 0, 0), dtype=np.uint8)
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
        return np.transpose(frames, (0, 3, 1, 2))  # -> NCHW

    def get_video_duration(self, video_path: str) -> float:
        vr = VideoReader(video_path, ctx=cpu(0))
        return len(vr) / vr.get_avg_fps()

