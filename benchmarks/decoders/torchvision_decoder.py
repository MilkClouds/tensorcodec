"""Benchmark adapters for torchvision.io.VideoReader.

Registers one adapter per available backend (``pyav``, ``video_reader``).
The benchmark registry in ``__init__.py`` catches the ``ImportError`` if
torchvision is missing.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import torchvision
from torchvision.io import VideoReader

# Suppress the deprecation warning during benchmarks
warnings.filterwarnings("ignore", message=".*video decoding.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Accurate seek.*")


class _TorchVisionDecoder:
    """Base adapter for :class:`torchvision.io.VideoReader`."""

    def __init__(self, backend: str) -> None:
        self._backend = backend

    def _open(self, video_path: str) -> VideoReader:
        torchvision.set_video_backend(self._backend)
        return VideoReader(video_path, "video")

    def get_frames_played_at(
        self, video_path: str, seconds: List[float]
    ) -> np.ndarray:
        vr = self._open(video_path)

        frames: list[np.ndarray] = []
        for ts in seconds:
            vr.seek(max(ts - 1.0, 0.0))  # seek before target (keyframe seek)
            best = None
            for frame in vr:
                if frame["pts"] > ts + 0.0001:
                    break
                best = frame
            if best is not None:
                frames.append(best["data"].numpy())  # already CHW
            else:
                # Fallback: seek to 0 and scan
                vr.seek(0.0)
                for frame in vr:
                    if frame["pts"] > ts + 0.0001:
                        break
                    best = frame
                if best is not None:
                    frames.append(best["data"].numpy())

        if not frames:
            return np.empty((0, 3, 0, 0), dtype=np.uint8)
        return np.stack(frames)  # NCHW

    def get_frames_played_in_range(
        self,
        video_path: str,
        start_seconds: float,
        stop_seconds: float,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        if fps is not None:
            raise ValueError("TorchVisionDecoder does not support the fps parameter")
        vr = self._open(video_path)
        if start_seconds > 0:
            vr.seek(start_seconds)

        frames: list[np.ndarray] = []
        for frame in vr:
            if frame["pts"] >= stop_seconds:
                break
            if frame["pts"] >= start_seconds:
                frames.append(frame["data"].numpy())  # CHW

        if not frames:
            return np.empty((0, 3, 0, 0), dtype=np.uint8)
        return np.stack(frames)  # NCHW

    def get_video_duration(self, video_path: str) -> float:
        vr = self._open(video_path)
        meta = vr.get_metadata()
        return float(meta["video"]["duration"][0])


class TorchVisionPyAVDecoder(_TorchVisionDecoder):
    """torchvision.io.VideoReader with the **pyav** backend."""

    name = "torchvision-pyav"

    def __init__(self) -> None:
        super().__init__("pyav")


class TorchVisionVideoReaderDecoder(_TorchVisionDecoder):
    """torchvision.io.VideoReader with the **video_reader** backend."""

    name = "torchvision-video_reader"

    def __init__(self) -> None:
        super().__init__("video_reader")


def available_backends() -> list[str]:
    """Return list of backends that actually work."""
    backends: list[str] = []
    for backend in ("pyav", "video_reader"):
        try:
            torchvision.set_video_backend(backend)
            backends.append(backend)
        except RuntimeError:
            pass
    return backends

