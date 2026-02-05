"""MP4/MOV container handler with optimized metadata reading.

MP4 containers store timing information in the moov atom:
- stts (sample-to-time): Frame durations (run-length encoded)
- ctts (composition time offset): DTS to PTS conversion for B-frames
- stss (sync sample): Keyframe indices
- stsz (sample size): Frame sizes

By reading these atoms directly, we can build the frame index without
scanning all packets, which is significantly faster for large files.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import BinaryIO, List, Optional, Set, Tuple, Union

from avdec._types import FrameIndex, FrameInfo
from avdec.containers.base import ContainerHandler

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# MP4 box type constants
BOX_MOOV = b"moov"
BOX_TRAK = b"trak"
BOX_MDIA = b"mdia"
BOX_MINF = b"minf"
BOX_STBL = b"stbl"
BOX_STTS = b"stts"
BOX_CTTS = b"ctts"
BOX_STSS = b"stss"
BOX_MDHD = b"mdhd"
BOX_HDLR = b"hdlr"


class MP4Handler(ContainerHandler):
    """Handler for MP4/MOV/M4V containers."""

    @property
    def name(self) -> str:
        return "MP4"

    @property
    def extensions(self) -> Set[str]:
        return {".mp4", ".m4v", ".mov", ".m4a"}

    def can_handle(self, source: PathLike) -> bool:
        """Check if file is a valid MP4 by reading ftyp box."""
        path = Path(source)
        if path.suffix.lower() not in self.extensions:
            return False

        try:
            with open(path, "rb") as f:
                # Read first 8 bytes to check for ftyp box
                header = f.read(8)
                if len(header) < 8:
                    return False
                size = struct.unpack(">I", header[:4])[0]
                box_type = header[4:8]
                # ftyp should be first box (or second after free/skip)
                if box_type == b"ftyp":
                    return True
                # Sometimes there's a free/skip box first
                if box_type in (b"free", b"skip", b"wide"):
                    f.seek(size)
                    header = f.read(8)
                    if len(header) >= 8 and header[4:8] == b"ftyp":
                        return True
            return False
        except (OSError, struct.error):
            return False

    def build_frame_index(
        self,
        source: PathLike,
        stream_index: Optional[int] = None,
    ) -> Optional[FrameIndex]:
        """Build frame index by parsing MP4 atoms directly."""
        path = Path(source)

        try:
            with open(path, "rb") as f:
                # Find moov box
                moov_data = self._find_and_read_box(f, BOX_MOOV)
                if moov_data is None:
                    logger.debug("moov box not found")
                    return None

                # Find video track
                track_data = self._find_video_track(moov_data, stream_index)
                if track_data is None:
                    logger.debug("Video track not found")
                    return None

                timescale, stbl_data = track_data

                # Parse timing tables
                frame_index = self._parse_stbl(stbl_data, timescale)
                return frame_index

        except Exception as e:
            logger.debug(f"MP4 parsing failed: {e}")
            return None

    def _find_and_read_box(
        self, f: BinaryIO, target_type: bytes, end_pos: Optional[int] = None
    ) -> Optional[bytes]:
        """Find and read a box by type."""
        if end_pos is None:
            f.seek(0, 2)  # Seek to end
            end_pos = f.tell()
            f.seek(0)

        while f.tell() < end_pos:
            pos = f.tell()
            header = f.read(8)
            if len(header) < 8:
                break

            size = struct.unpack(">I", header[:4])[0]
            box_type = header[4:8]

            if size == 0:  # Box extends to end of file
                size = end_pos - pos
            elif size == 1:  # 64-bit size
                size = struct.unpack(">Q", f.read(8))[0]

            if box_type == target_type:
                # Read box content (excluding header)
                content_size = size - 8
                if content_size > 100_000_000:  # Sanity check: 100MB max
                    return None
                return f.read(content_size)

            # Skip to next box
            f.seek(pos + size)

        return None

    def _find_box_in_data(
        self, data: bytes, target_type: bytes, offset: int = 0
    ) -> Optional[Tuple[int, bytes]]:
        """Find a box within already-read data."""
        pos = offset
        while pos < len(data) - 8:
            size = struct.unpack(">I", data[pos : pos + 4])[0]
            box_type = data[pos + 4 : pos + 8]

            if size == 0:
                size = len(data) - pos
            elif size == 1 and pos + 16 <= len(data):
                size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]

            if size < 8 or pos + size > len(data):
                break

            if box_type == target_type:
                content_start = pos + 8
                content_end = pos + size
                return (pos, data[content_start:content_end])

            pos += size

        return None

    def _find_video_track(
        self, moov_data: bytes, stream_index: Optional[int]
    ) -> Optional[Tuple[int, bytes]]:
        """Find video track and return (timescale, stbl_data)."""
        pos = 0
        track_idx = 0

        while pos < len(moov_data):
            result = self._find_box_in_data(moov_data, BOX_TRAK, pos)
            if result is None:
                break

            trak_pos, trak_data = result
            pos = trak_pos + 8 + len(trak_data)

            # Check if this is a video track by looking at hdlr
            mdia_result = self._find_box_in_data(trak_data, BOX_MDIA)
            if mdia_result is None:
                continue

            _, mdia_data = mdia_result

            # Check handler type
            hdlr_result = self._find_box_in_data(mdia_data, BOX_HDLR)
            if hdlr_result is None:
                continue

            _, hdlr_data = hdlr_result
            if len(hdlr_data) < 12:
                continue

            # hdlr: version(1) + flags(3) + pre_defined(4) + handler_type(4)
            handler_type = hdlr_data[8:12]
            if handler_type != b"vide":
                continue

            # This is a video track
            if stream_index is not None and track_idx != stream_index:
                track_idx += 1
                continue

            # Get timescale from mdhd
            mdhd_result = self._find_box_in_data(mdia_data, BOX_MDHD)
            if mdhd_result is None:
                continue

            _, mdhd_data = mdhd_result
            timescale = self._parse_mdhd_timescale(mdhd_data)
            if timescale is None:
                continue

            # Get stbl
            minf_result = self._find_box_in_data(mdia_data, BOX_MINF)
            if minf_result is None:
                continue

            _, minf_data = minf_result
            stbl_result = self._find_box_in_data(minf_data, BOX_STBL)
            if stbl_result is None:
                continue

            _, stbl_data = stbl_result
            return (timescale, stbl_data)

        return None

    def _parse_mdhd_timescale(self, mdhd_data: bytes) -> Optional[int]:
        """Parse timescale from mdhd box."""
        if len(mdhd_data) < 4:
            return None

        version = mdhd_data[0]
        if version == 0:
            # 32-bit: version(1) + flags(3) + creation(4) + modification(4) + timescale(4)
            if len(mdhd_data) < 16:
                return None
            return struct.unpack(">I", mdhd_data[12:16])[0]
        elif version == 1:
            # 64-bit: version(1) + flags(3) + creation(8) + modification(8) + timescale(4)
            if len(mdhd_data) < 24:
                return None
            return struct.unpack(">I", mdhd_data[20:24])[0]

        return None

    def _parse_stbl(self, stbl_data: bytes, timescale: int) -> Optional[FrameIndex]:
        """Parse stbl box to build frame index."""
        # Parse stts (sample-to-time table)
        stts_result = self._find_box_in_data(stbl_data, BOX_STTS)
        if stts_result is None:
            return None

        _, stts_data = stts_result
        durations = self._parse_stts(stts_data)
        if not durations:
            return None

        # Parse ctts (composition time offset) - optional
        ctts_result = self._find_box_in_data(stbl_data, BOX_CTTS)
        ctts_offsets: List[int] = []
        if ctts_result is not None:
            _, ctts_data = ctts_result
            ctts_offsets = self._parse_ctts(ctts_data)

        # Parse stss (sync sample / keyframes) - optional
        stss_result = self._find_box_in_data(stbl_data, BOX_STSS)
        keyframe_set: set = set()
        if stss_result is not None:
            _, stss_data = stss_result
            keyframe_set = set(self._parse_stss(stss_data))

        # Build frame infos
        frame_infos: List[FrameInfo] = []
        keyframe_indices: List[int] = []

        dts = 0  # Decode timestamp in timescale units
        for i, duration in enumerate(durations):
            # PTS = DTS + CTS offset
            cts_offset = ctts_offsets[i] if i < len(ctts_offsets) else 0
            pts = dts + cts_offset
            pts_seconds = pts / timescale

            # 1-indexed in stss, so check i+1
            is_keyframe = (i + 1) in keyframe_set or len(keyframe_set) == 0

            frame_infos.append(
                FrameInfo(
                    pts=pts_seconds,
                    next_pts=float("inf"),
                    frame_index=i,
                    is_keyframe=is_keyframe,
                )
            )

            if is_keyframe:
                keyframe_indices.append(i)

            dts += duration

        # Sort by PTS and update next_pts
        frame_infos.sort(key=lambda f: f.pts)

        # Normalize PTS to start at 0 (handle edit list / ctts offset)
        # This matches PyAV/FFmpeg behavior where first frame PTS is 0
        if frame_infos:
            min_pts = frame_infos[0].pts
            if min_pts > 0:
                for info in frame_infos:
                    info.pts -= min_pts

        # Update frame indices and next_pts after sorting/normalization
        for i, info in enumerate(frame_infos):
            info.frame_index = i
            if i + 1 < len(frame_infos):
                info.next_pts = frame_infos[i + 1].pts

        # Rebuild keyframe indices after sorting
        keyframe_indices = [i for i, info in enumerate(frame_infos) if info.is_keyframe]

        return FrameIndex(frame_infos=frame_infos, keyframe_indices=keyframe_indices)

    def _parse_stts(self, stts_data: bytes) -> List[int]:
        """Parse stts box to get frame durations."""
        if len(stts_data) < 8:
            return []

        # version(1) + flags(3) + entry_count(4)
        entry_count = struct.unpack(">I", stts_data[4:8])[0]

        durations: List[int] = []
        offset = 8
        for _ in range(entry_count):
            if offset + 8 > len(stts_data):
                break
            sample_count = struct.unpack(">I", stts_data[offset : offset + 4])[0]
            sample_delta = struct.unpack(">I", stts_data[offset + 4 : offset + 8])[0]
            durations.extend([sample_delta] * sample_count)
            offset += 8

        return durations

    def _parse_ctts(self, ctts_data: bytes) -> List[int]:
        """Parse ctts box to get composition time offsets."""
        if len(ctts_data) < 8:
            return []

        version = ctts_data[0]
        entry_count = struct.unpack(">I", ctts_data[4:8])[0]

        offsets: List[int] = []
        offset = 8
        for _ in range(entry_count):
            if offset + 8 > len(ctts_data):
                break
            sample_count = struct.unpack(">I", ctts_data[offset : offset + 4])[0]
            if version == 0:
                sample_offset = struct.unpack(">I", ctts_data[offset + 4 : offset + 8])[0]
            else:
                sample_offset = struct.unpack(">i", ctts_data[offset + 4 : offset + 8])[0]
            offsets.extend([sample_offset] * sample_count)
            offset += 8

        return offsets

    def _parse_stss(self, stss_data: bytes) -> List[int]:
        """Parse stss box to get keyframe indices (1-indexed)."""
        if len(stss_data) < 8:
            return []

        entry_count = struct.unpack(">I", stss_data[4:8])[0]

        keyframes: List[int] = []
        offset = 8
        for _ in range(entry_count):
            if offset + 4 > len(stss_data):
                break
            sample_number = struct.unpack(">I", stss_data[offset : offset + 4])[0]
            keyframes.append(sample_number)
            offset += 4

        return keyframes


__all__ = ["MP4Handler"]

