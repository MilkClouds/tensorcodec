"""Benchmark adapter for TorchCodec.

Requires ``torchcodec`` to be installed.  The benchmark registry in
``__init__.py`` catches the ``ImportError`` if it is missing.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from torchcodec.decoders import VideoDecoder as TorchCodecVideoDecoder


class TorchCodecDecoder:
    """Wraps :class:`torchcodec.decoders.VideoDecoder` for the benchmark protocol."""

    name = "torchcodec"

    def get_frames_played_at(
        self, video_path: str, seconds: List[float]
    ) -> np.ndarray:
        dec = TorchCodecVideoDecoder(video_path)
        batch = dec.get_frames_played_at(seconds)
        return batch.data.numpy()  # NCHW uint8

    def get_frames_played_in_range(
        self,
        video_path: str,
        start_seconds: float,
        stop_seconds: float,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        dec = TorchCodecVideoDecoder(video_path)
        batch = dec.get_frames_played_in_range(start_seconds, stop_seconds, fps=fps)
        return batch.data.numpy()  # NCHW uint8

    def get_video_duration(self, video_path: str) -> float:
        dec = TorchCodecVideoDecoder(video_path)
        return float(dec.metadata.end_stream_seconds)

