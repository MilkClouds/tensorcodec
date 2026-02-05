"""Type definitions for avdec.

# =============================================================================
# TorchCodec Compatibility Notes
# =============================================================================
# This module defines types that match TorchCodec's interfaces:
#
#   - SeekMode               -> VideoDecoder(seek_mode=...) parameter
#   - DimensionOrder         -> VideoDecoder(dimension_order=...) parameter
#   - FrameInfo              -> TorchCodec's internal FrameInfo struct
#   - VideoStreamMetadata    -> VideoDecoder.metadata return type
#   - Frame                  -> Return type of get_frame_at() / get_frame_played_at()
#   - FrameBatch             -> Return type of get_frames_at() / get_frames_played_at()
#   - FrameIndex             -> TorchCodec's allFrames vector (exact mode)
#
# DO NOT modify these structures without verifying TorchCodec compatibility.
# =============================================================================
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

# Fraction is used in VideoStreamMetadata.time_base
from fractions import Fraction  # noqa: F401

# Type alias for dimension order
DimensionOrder = Literal["NCHW", "NHWC"]


class SeekMode(str, enum.Enum):
    """Seek mode for video decoding.

    [TorchCodec Compatibility: VideoDecoder(seek_mode=...)]

    - exact: Scan all packets to build frame index (accurate, slower initial load)
    - approximate: Use average FPS to estimate frame positions (fast, less accurate for VFR)
    """

    EXACT = "exact"
    APPROXIMATE = "approximate"


@dataclass
class FrameInfo:
    """Information about a single video frame.

    [TorchCodec Compatibility: Internal FrameInfo struct]
    This matches TorchCodec's internal FrameInfo structure used in
    VideoDecoder.cpp's allFrames vector.

    Attributes:
        pts: Presentation timestamp in seconds
        next_pts: PTS of the next frame (float('inf') for last frame)
            - Used for playback semantics: frame[i].pts <= t < frame[i].next_pts
        frame_index: Zero-based frame index
        is_keyframe: Whether this frame is a keyframe
    """

    pts: float
    next_pts: float
    frame_index: int
    is_keyframe: bool


@dataclass
class VideoStreamMetadata:
    """Video stream metadata container.

    [TorchCodec Compatibility: VideoDecoder.metadata return type]
    Matches the structure returned by TorchCodec's VideoDecoder.metadata property.

    Attributes:
        num_frames: Number of frames (from content scan if available, else header)
        duration_seconds: Video duration in seconds (computed with fallback logic)
        average_fps: Average frame rate (computed with fallback logic)
        width: Frame width in pixels
        height: Frame height in pixels
        codec: Codec name (e.g., 'h264', 'hevc')
        time_base: Stream time base as Fraction

        # TorchCodec extended fields
        begin_stream_seconds: First frame PTS in seconds (0 if unknown)
        end_stream_seconds: Last frame PTS + duration in seconds
        num_frames_from_header: Frame count from container header (may be inaccurate)
        num_frames_from_content: Frame count from packet scan (accurate, None if not scanned)
        average_fps_from_header: FPS from header (may be inaccurate)
        duration_seconds_from_header: Duration from header (may be inaccurate)
    """

    num_frames: int
    duration_seconds: float
    average_fps: float
    width: int
    height: int
    codec: Optional[str] = None
    time_base: Optional[Fraction] = None

    # TorchCodec extended fields
    begin_stream_seconds: float = 0.0
    end_stream_seconds: Optional[float] = None
    num_frames_from_header: Optional[int] = None
    num_frames_from_content: Optional[int] = None
    average_fps_from_header: Optional[float] = None
    duration_seconds_from_header: Optional[float] = None


@dataclass
class Frame:
    """Single decoded video frame.

    [TorchCodec Compatibility: Return type of get_frame_at() / get_frame_played_at()]
    This structure matches TorchCodec's Frame dataclass.

    Attributes:
        data: Frame data as numpy array (H, W, C) or (C, H, W) depending on dimension_order
        pts_seconds: Presentation timestamp in seconds
        duration_seconds: Duration of the frame in seconds
    """

    data: npt.NDArray[np.uint8]
    pts_seconds: float
    duration_seconds: float

    def __repr__(self) -> str:
        """Return string representation."""
        shape = self.data.shape
        return f"Frame(pts={self.pts_seconds:.3f}s, shape={shape})"


@dataclass
class FrameBatch:
    """Batch of decoded video frames.

    [TorchCodec Compatibility: Return type of get_frames_at() / get_frames_played_at()]
    This structure matches the return type of TorchCodec's batch frame retrieval methods.

    Attributes:
        data: Frame data as numpy array in NHWC format (N, H, W, C) or NCHW (N, C, H, W)
              where C=3 for RGB
        pts_seconds: Presentation timestamps in seconds for each frame
        duration_seconds: Duration of each frame in seconds
        frame_indices: Frame indices for each frame (avdec extension, not in TorchCodec)
    """

    data: npt.NDArray[np.uint8]
    pts_seconds: npt.NDArray[np.float64]
    duration_seconds: npt.NDArray[np.float64]
    frame_indices: Optional[npt.NDArray[np.int64]] = None

    def __len__(self) -> int:
        """Return number of frames in batch."""
        return len(self.data)

    def __getitem__(self, idx: Union[int, slice]) -> npt.NDArray[np.uint8]:
        """Get frame(s) by index or slice."""
        return self.data[idx]

    def __repr__(self) -> str:
        """Return string representation."""
        n = len(self.data)
        if n > 0:
            shape = self.data.shape[1:]
            return f"FrameBatch(n={n}, shape={shape})"
        return "FrameBatch(n=0)"


@dataclass
class FrameIndex:
    """Pre-built frame index for exact mode decoding.

    [TorchCodec Compatibility: allFrames vector]
    This corresponds to TorchCodec's VideoDecoder.cpp allFrames vector,
    which stores all frame PTS values built during the initial packet scan.

    Stores all frame PTS values and keyframe positions for efficient
    timestamp-to-frame-index conversion using bisect_right.

    Attributes:
        frame_infos: List of FrameInfo for each frame
        keyframe_indices: Indices of keyframes in frame_infos
    """

    frame_infos: List[FrameInfo]
    keyframe_indices: List[int]

    @property
    def pts_list(self) -> List[float]:
        """Get list of PTS values for all frames."""
        return [f.pts for f in self.frame_infos]

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.frame_infos)


__all__ = [
    "DimensionOrder",
    "SeekMode",
    "FrameInfo",
    "VideoStreamMetadata",
    "Frame",
    "FrameBatch",
    "FrameIndex",
]

