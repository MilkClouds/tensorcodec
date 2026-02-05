"""Metadata readers for video containers.

Provides fast metadata reading for supported containers (MP4) and
fallback packet scanning for others.

# =============================================================================
# TorchCodec Compatibility Notes
# =============================================================================
# This module implements the frame index building that corresponds to
# TorchCodec's "exact mode" (seek_mode="exact") initialization.
#
# Key functions:
#   - build_frame_index_from_scan() -> TorchCodec's allFrames vector building
#   - get_frame_index()             -> Entry point for frame index construction
#
# The frame index stores (pts, next_pts) pairs for each frame, enabling
# TorchCodec's playback semantics: frame[i].pts <= timestamp < frame[i+1].pts
# =============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from avdec._types import FrameIndex, FrameInfo

if TYPE_CHECKING:
    import av

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _try_container_optimized_read(
    source: PathLike,
    container: "av.InputContainer",
    stream_index: Optional[int] = None,
) -> Optional[FrameIndex]:
    """Try to read frame index using container-specific optimization.

    Uses the modular container handler system for format-specific
    optimizations (e.g., direct MP4 atom parsing).

    Args:
        source: Path to video file
        container: Open PyAV container
        stream_index: Video stream index

    Returns:
        FrameIndex if successful, None to fall back to packet scan
    """
    try:
        # Import here to avoid circular imports and allow lazy loading
        from avdec.containers import get_handler
    except ImportError:
        logger.debug("Container handlers not available")
        return None

    handler = get_handler(source)
    if handler is None:
        return None

    try:
        frame_index = handler.build_frame_index(source, stream_index)
        if frame_index is not None:
            logger.debug(f"Used {handler.name} optimization for {source}")
            return frame_index
    except Exception as e:
        logger.debug(f"{handler.name} optimization failed: {e}")

    return None


def build_frame_index_from_scan(
    container: "av.InputContainer",
    stream_index: Optional[int] = None,
) -> FrameIndex:
    """Build frame index by scanning all packets.

    [TorchCodec Compatibility: Exact mode packet scan]
    This implements TorchCodec's exact mode initialization where all packets
    are scanned to build the allFrames vector with accurate PTS values.

    Reference: TorchCodec's VideoDecoder.cpp, scanFileAndUpdateMetadataAndIndex()

    This is the fallback method that works for all containers but requires
    reading all packets (I/O intensive).

    Args:
        container: Open PyAV container
        stream_index: Absolute stream index in container. If None, uses first video stream.

    Returns:
        FrameIndex with all frame timing information
    """
    # Find the correct stream
    if stream_index is not None:
        stream = None
        for s in container.streams:
            if s.index == stream_index:
                stream = s
                break
        if stream is None:
            raise ValueError(f"Stream index {stream_index} not found")
    else:
        stream = container.streams.video[0]

    time_base = float(stream.time_base)

    frame_infos: List[FrameInfo] = []
    keyframe_indices: List[int] = []

    # Demux packets to get PTS without decoding
    # [TorchCodec Compatibility: getPtsOrDts() fallback]
    # TorchCodec uses getPtsOrDts() which falls back to DTS if PTS is invalid
    for idx, packet in enumerate(container.demux(stream)):
        # Use PTS if available, fall back to DTS (TorchCodec's getPtsOrDts())
        pts = packet.pts
        if pts is None:
            pts = packet.dts
        if pts is None:
            continue

        pts_seconds = pts * time_base
        is_keyframe = packet.is_keyframe

        frame_infos.append(
            FrameInfo(
                pts=pts_seconds,
                next_pts=float("inf"),  # Will be filled in later
                frame_index=idx,
                is_keyframe=is_keyframe,
            )
        )

        if is_keyframe:
            keyframe_indices.append(idx)

    # Sort by PTS (packets may not be in presentation order for B-frames)
    frame_infos.sort(key=lambda f: f.pts)

    # Update frame indices and next_pts after sorting
    for i, info in enumerate(frame_infos):
        info.frame_index = i
        if i + 1 < len(frame_infos):
            info.next_pts = frame_infos[i + 1].pts

    # Rebuild keyframe_indices after sorting
    keyframe_indices = [i for i, info in enumerate(frame_infos) if info.is_keyframe]

    # Seek back to beginning
    container.seek(0)

    return FrameIndex(frame_infos=frame_infos, keyframe_indices=keyframe_indices)


def get_frame_index(
    source: PathLike,
    container: "av.InputContainer",
    stream_index: Optional[int] = None,
    *,
    force_scan: bool = False,
) -> FrameIndex:
    """Get frame index for a video source.

    Tries container-specific optimizations first, falls back to packet scan.

    Args:
        source: Path to video file
        container: Open PyAV container (used for fallback scan)
        stream_index: Absolute stream index in container. If None, uses first video stream.
        force_scan: If True, skip optimizations and always scan packets

    Returns:
        FrameIndex with all frame timing information
    """
    if not force_scan:
        # Try container-specific optimization (MP4, MKV, etc.)
        frame_index = _try_container_optimized_read(source, container, stream_index)
        if frame_index is not None:
            return frame_index

    # Fallback to packet scan
    logger.debug(f"Using packet scan for {source}")
    return build_frame_index_from_scan(container, stream_index)


__all__ = [
    "get_frame_index",
    "build_frame_index_from_scan",
]

