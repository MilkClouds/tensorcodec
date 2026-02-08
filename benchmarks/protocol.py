"""Protocol and result types for decoder benchmarks.

A decoder only needs to satisfy :class:`VideoDecoderProtocol` to be
benchmarked. The protocol mirrors avdec's public API:

    - ``get_frames_played_at``
    - ``get_frames_played_in_range``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class VideoDecoderProtocol(Protocol):
    """Minimal interface a decoder must implement to be benchmarked.

    Every method receives the *video path* so decoders can be stateless
    (open-decode-close per call) or keep state — the benchmark does not care.
    """

    name: str

    def get_frames_played_at(self, video_path: str, seconds: List[float]) -> np.ndarray:
        """Return NCHW uint8 array for the given timestamps."""
        ...

    def get_frames_played_in_range(
        self,
        video_path: str,
        start_seconds: float,
        stop_seconds: float,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Return NCHW uint8 array for frames in ``[start, stop)``."""
        ...

    def get_video_duration(self, video_path: str) -> float:
        """Return video duration in seconds (used to generate queries)."""
        ...


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    """Outcome of one benchmark scenario."""

    decoder_name: str
    scenario: str
    num_frames: int
    elapsed_time: float
    io_bytes: Optional[int] = None
    io_calls: Optional[int] = None
    timed_out: bool = False
    num_runs_completed: int = 1

    @property
    def fps(self) -> float:
        return self.num_frames / self.elapsed_time if self.elapsed_time > 0 else 0.0

    @property
    def bytes_per_frame(self) -> Optional[float]:
        if self.io_bytes is None:
            return None
        return self.io_bytes / self.num_frames if self.num_frames > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder": self.decoder_name,
            "scenario": self.scenario,
            "frames": self.num_frames,
            "elapsed_s": round(self.elapsed_time, 4),
            "fps": round(self.fps, 1),
            "io_bytes": self.io_bytes,
            "io_calls": self.io_calls,
            "bytes_per_frame": round(self.bytes_per_frame, 1) if self.bytes_per_frame is not None else None,
            "timed_out": self.timed_out,
        }
