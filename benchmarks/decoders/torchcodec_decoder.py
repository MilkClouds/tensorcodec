"""Benchmark adapter for TorchCodec.

Requires ``torchcodec`` to be installed.  The benchmark registry in
``__init__.py`` catches the ``ImportError`` if it is missing.

Multiple configurations are exposed so the benchmark can compare
``seek_mode`` and ``num_ffmpeg_threads`` settings.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from torchcodec.decoders import VideoDecoder as TorchCodecVideoDecoder
from torchcodec.decoders import set_cuda_backend


class TorchCodecDecoder:
    """Wraps :class:`torchcodec.decoders.VideoDecoder` for the benchmark protocol.

    Parameters
    ----------
    seek_mode : ``"exact"`` | ``"approximate"``
        Passed to ``VideoDecoder(..., seek_mode=...)``.
    num_ffmpeg_threads : int
        Passed to ``VideoDecoder(..., num_ffmpeg_threads=...)``.
    name : str | None
        Display name for benchmark output.  Auto-generated if *None*.
    """

    def __init__(
        self,
        seek_mode: str = "exact",
        num_ffmpeg_threads: int = 1,
        name: str | None = None,
    ) -> None:
        self.seek_mode = seek_mode
        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.name = name or f"torchcodec(seek={seek_mode},thr={num_ffmpeg_threads})"

    def _open(self, video_path: str) -> TorchCodecVideoDecoder:
        return TorchCodecVideoDecoder(
            video_path,
            seek_mode=self.seek_mode,
            num_ffmpeg_threads=self.num_ffmpeg_threads,
        )

    def get_frames_played_at(self, video_path: str, seconds: List[float]) -> np.ndarray:
        dec = self._open(video_path)
        batch = dec.get_frames_played_at(seconds)
        return batch.data.numpy()  # NCHW uint8

    def get_frames_played_in_range(
        self,
        video_path: str,
        start_seconds: float,
        stop_seconds: float,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        if fps is not None:
            raise ValueError("TorchCodecDecoder does not support the fps parameter")
        dec = self._open(video_path)
        batch = dec.get_frames_played_in_range(start_seconds, stop_seconds)
        return batch.data.numpy()  # NCHW uint8

    def get_video_duration(self, video_path: str) -> float:
        dec = self._open(video_path)
        return float(dec.metadata.end_stream_seconds)


class TorchCodecGPUDecoder(TorchCodecDecoder):
    """GPU-accelerated variant using ``device="cuda"``.

    Decoded frames live on the GPU; they are moved to CPU/NumPy only at
    the boundary so the benchmark protocol (which expects ``np.ndarray``)
    is satisfied.
    """

    def __init__(
        self,
        seek_mode: str = "approximate",
        num_ffmpeg_threads: int = 1,
        name: str | None = None,
    ) -> None:
        super().__init__(
            seek_mode=seek_mode,
            num_ffmpeg_threads=num_ffmpeg_threads,
            name=name or f"torchcodec(gpu,seek={seek_mode})",
        )

    def _open(self, video_path: str) -> TorchCodecVideoDecoder:
        # NOTE: beta backend present at torchcodec>=0.8.0
        with set_cuda_backend("beta"):
            return TorchCodecVideoDecoder(
                video_path,
                device="cuda",
                seek_mode=self.seek_mode,
                num_ffmpeg_threads=self.num_ffmpeg_threads,
            )

    def get_frames_played_at(self, video_path: str, seconds: List[float]) -> np.ndarray:
        dec = self._open(video_path)
        batch = dec.get_frames_played_at(seconds)
        return batch.data.cpu().numpy()  # GPU → CPU → NumPy

    def get_frames_played_in_range(
        self,
        video_path: str,
        start_seconds: float,
        stop_seconds: float,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        if fps is not None:
            raise ValueError("TorchCodecGPUDecoder does not support the fps parameter")
        dec = self._open(video_path)
        batch = dec.get_frames_played_in_range(start_seconds, stop_seconds)
        return batch.data.cpu().numpy()  # GPU → CPU → NumPy


# Pre-built configurations for the benchmark registry
TORCHCODEC_CONFIGS: list[TorchCodecDecoder] = [
    TorchCodecDecoder(seek_mode="exact", num_ffmpeg_threads=1),
    TorchCodecDecoder(seek_mode="exact", num_ffmpeg_threads=0),
    TorchCodecDecoder(seek_mode="approximate", num_ffmpeg_threads=1),
    TorchCodecDecoder(seek_mode="approximate", num_ffmpeg_threads=0),
]

# GPU configs — only instantiated when CUDA is available (checked in __init__.py)
TORCHCODEC_GPU_CONFIGS: list[TorchCodecGPUDecoder] = [
    TorchCodecGPUDecoder(seek_mode="approximate"),
    TorchCodecGPUDecoder(seek_mode="exact"),
]
