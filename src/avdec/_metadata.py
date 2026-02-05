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


def _try_mp4_fast_read(source: PathLike) -> Optional[FrameIndex]:
    """Try to read MP4 metadata using pymp4 (stts/ctts atoms).

    This is much faster than scanning all packets because it only reads
    the moov atom which contains all timing information.

    Args:
        source: Path to video file

    Returns:
        FrameIndex if successful, None if pymp4 not available or not MP4
    """
    try:
        import pymp4.parser  # noqa: F401
    except ImportError:
        logger.debug("pymp4 not installed, falling back to packet scan")
        return None

    source_path = Path(source)
    if source_path.suffix.lower() not in (".mp4", ".m4v", ".mov"):
        return None

    try:
        return _parse_mp4_metadata(source_path)
    except Exception as e:
        logger.debug(f"MP4 fast read failed: {e}, falling back to packet scan")
        return None


def _parse_mp4_metadata(source: Path) -> Optional[FrameIndex]:
    """Parse MP4 stts/ctts atoms to build frame index.

    MP4 containers store timing in:
    - stts: sample-to-time table (run-length encoded durations)
    - ctts: composition time offset (DTS -> PTS conversion)
    - stss: sync sample table (keyframe indices)
    """
    # Placeholder - full implementation requires traversing box hierarchy
    # For now, return None to use fallback packet scan
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
    for idx, packet in enumerate(container.demux(stream)):
        if packet.pts is None:
            continue

        pts_seconds = packet.pts * time_base
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

    Tries fast metadata reading first (for MP4), falls back to packet scan.

    Args:
        source: Path to video file
        container: Open PyAV container (used for fallback scan)
        stream_index: Absolute stream index in container. If None, uses first video stream.
        force_scan: If True, skip fast read and always scan packets

    Returns:
        FrameIndex with all frame timing information
    """
    if not force_scan:
        # Try fast MP4 metadata read
        frame_index = _try_mp4_fast_read(source)
        if frame_index is not None:
            logger.debug(f"Used fast MP4 metadata read for {source}")
            return frame_index

    # Fallback to packet scan
    logger.debug(f"Using packet scan for {source}")
    return build_frame_index_from_scan(container, stream_index)


__all__ = [
    "get_frame_index",
    "build_frame_index_from_scan",
]

