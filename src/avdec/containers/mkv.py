"""MKV/WebM container handler with optimized metadata reading.

MKV (Matroska) containers use EBML (Extensible Binary Meta Language) format.
Key elements for frame indexing:
- Segment > Info > TimestampScale: Base timestamp unit (default 1000000 = 1ms)
- Segment > Tracks > TrackEntry: Track metadata including CodecID
- Segment > Cues > CuePoint: Keyframe positions and timestamps

The Cues element provides direct access to keyframe positions, enabling
fast seeking without scanning all clusters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO, List, Optional, Set, Tuple, Union

from avdec._types import FrameIndex
from avdec.containers.base import ContainerHandler

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# EBML Element IDs (variable length, shown as bytes)
EBML_HEADER = b"\x1a\x45\xdf\xa3"
SEGMENT = b"\x18\x53\x80\x67"
INFO = b"\x15\x49\xa9\x66"
TIMESTAMP_SCALE = b"\x2a\xd7\xb1"
TRACKS = b"\x16\x54\xae\x6b"
TRACK_ENTRY = b"\xae"
TRACK_NUMBER = b"\xd7"
TRACK_TYPE = b"\x83"
CUES = b"\x1c\x53\xbb\x6b"
CUE_POINT = b"\xbb"
CUE_TIME = b"\xb3"
CUE_TRACK_POSITIONS = b"\xb7"
CUE_TRACK = b"\xf7"
CUE_CLUSTER_POSITION = b"\xf1"

# Track types
TRACK_TYPE_VIDEO = 1


class MKVHandler(ContainerHandler):
    """Handler for MKV/WebM containers."""

    @property
    def name(self) -> str:
        return "MKV"

    @property
    def extensions(self) -> Set[str]:
        return {".mkv", ".webm", ".mka", ".mk3d"}

    def can_handle(self, source: PathLike) -> bool:
        """Check if file is a valid MKV by reading EBML header."""
        path = Path(source)
        if path.suffix.lower() not in self.extensions:
            return False

        try:
            with open(path, "rb") as f:
                # EBML header starts with 0x1A45DFA3
                header = f.read(4)
                return header == EBML_HEADER
        except OSError:
            return False

    def build_frame_index(
        self,
        source: PathLike,
        stream_index: Optional[int] = None,
    ) -> Optional[FrameIndex]:
        """Build frame index from MKV Cues element.
        
        Note: MKV Cues only contain keyframe positions, not all frames.
        For complete frame index, we still need packet scan.
        This method returns None to use fallback, but provides
        keyframe positions for optimized seeking.
        """
        # MKV Cues only have keyframes, not all frames
        # Return None to use packet scan for complete frame index
        # But we can still use Cues for seek optimization
        return None

    def get_keyframe_positions(
        self,
        source: PathLike,
        stream_index: Optional[int] = None,
    ) -> Optional[List[int]]:
        """Get keyframe byte positions from Cues element."""
        path = Path(source)

        try:
            with open(path, "rb") as f:
                # Find Segment
                if not self._skip_to_element(f, SEGMENT):
                    return None

                segment_start = f.tell()

                # Find Cues within Segment
                # Note: Cues may be at the end of file, this is a simplified search
                cues_data = self._find_element_data(f, CUES, max_search=50_000_000)
                if cues_data is None:
                    return None

                # Parse Cues to get cluster positions
                positions = self._parse_cues(cues_data, segment_start, stream_index)
                return positions if positions else None

        except Exception as e:
            logger.debug(f"MKV Cues parsing failed: {e}")
            return None

    def _read_vint(self, f: BinaryIO) -> Optional[Tuple[int, int]]:
        """Read EBML variable-length integer. Returns (value, byte_length)."""
        first = f.read(1)
        if not first:
            return None

        first_byte = first[0]
        if first_byte == 0:
            return None

        # Count leading zeros to determine length
        length = 1
        mask = 0x80
        while not (first_byte & mask) and length < 8:
            length += 1
            mask >>= 1

        # Read remaining bytes
        remaining = f.read(length - 1)
        if len(remaining) < length - 1:
            return None

        # Combine bytes
        value = first_byte & (mask - 1)  # Remove length marker
        for b in remaining:
            value = (value << 8) | b

        return (value, length)

    def _read_element_id(self, f: BinaryIO) -> Optional[bytes]:
        """Read EBML element ID."""
        first = f.read(1)
        if not first:
            return None

        first_byte = first[0]

        # Determine ID length from leading bits
        if first_byte & 0x80:
            return first
        elif first_byte & 0x40:
            remaining = f.read(1)
            return first + remaining if remaining else None
        elif first_byte & 0x20:
            remaining = f.read(2)
            return first + remaining if len(remaining) == 2 else None
        elif first_byte & 0x10:
            remaining = f.read(3)
            return first + remaining if len(remaining) == 3 else None
        else:
            return None

    def _skip_to_element(self, f: BinaryIO, target_id: bytes) -> bool:
        """Skip forward until we find the target element ID."""
        max_pos = f.tell() + 50_000_000  # Limit search to 50MB

        while f.tell() < max_pos:
            element_id = self._read_element_id(f)
            if element_id is None:
                return False

            if element_id == target_id:
                # Read and skip the size, position file at element data
                size_result = self._read_vint(f)
                if size_result is None:
                    return False
                return True

            # Read size and skip element
            size_result = self._read_vint(f)
            if size_result is None:
                return False

            size, _ = size_result
            # Skip element data
            f.seek(f.tell() + size)

        return False

    def _find_element_data(
        self, f: BinaryIO, target_id: bytes, max_search: int = 10_000_000
    ) -> Optional[bytes]:
        """Find element and return its data."""
        start_pos = f.tell()
        max_pos = start_pos + max_search

        while f.tell() < max_pos:
            element_id = self._read_element_id(f)
            if element_id is None:
                return None

            size_result = self._read_vint(f)
            if size_result is None:
                return None

            size, _ = size_result

            if element_id == target_id:
                if size > 100_000_000:  # Sanity check: 100MB max
                    return None
                return f.read(size)

            # Skip this element
            f.seek(f.tell() + size)

        return None

    def _parse_cues(
        self, cues_data: bytes, segment_start: int, stream_index: Optional[int]
    ) -> List[int]:
        """Parse Cues element to extract cluster positions."""
        positions: List[int] = []
        pos = 0

        while pos < len(cues_data):
            # Look for CuePoint
            if pos + 1 > len(cues_data):
                break

            # Simple search for CuePoint element (0xBB)
            if cues_data[pos : pos + 1] != CUE_POINT:
                pos += 1
                continue

            pos += 1  # Skip element ID

            # Read size
            size, size_len = self._read_vint_from_bytes(cues_data, pos)
            if size is None:
                break
            pos += size_len

            cue_end = pos + size
            cluster_pos = None

            # Parse CuePoint contents
            while pos < cue_end:
                if cues_data[pos : pos + 1] == CUE_TIME:
                    pos += 1
                    vint_size, vint_len = self._read_vint_from_bytes(cues_data, pos)
                    if vint_size is None:
                        break
                    pos += vint_len + vint_size
                elif cues_data[pos : pos + 1] == CUE_TRACK_POSITIONS:
                    pos += 1
                    vint_size, vint_len = self._read_vint_from_bytes(cues_data, pos)
                    if vint_size is None:
                        break
                    pos += vint_len

                    # Parse CueTrackPositions
                    track_pos_end = pos + vint_size
                    while pos < track_pos_end:
                        if cues_data[pos : pos + 1] == CUE_CLUSTER_POSITION:
                            pos += 1
                            vint_size2, vint_len2 = self._read_vint_from_bytes(
                                cues_data, pos
                            )
                            if vint_size2 is None:
                                break
                            pos += vint_len2
                            # Read cluster position value
                            cluster_pos = self._read_uint_from_bytes(
                                cues_data, pos, vint_size2
                            )
                            pos += vint_size2
                        else:
                            # Skip other elements
                            pos += 1
                            vint_size2, vint_len2 = self._read_vint_from_bytes(
                                cues_data, pos
                            )
                            if vint_size2 is None:
                                break
                            pos += vint_len2 + vint_size2
                else:
                    pos += 1

            if cluster_pos is not None:
                # Convert relative position to absolute
                positions.append(segment_start + cluster_pos)

            pos = cue_end

        return positions

    def _read_vint_from_bytes(
        self, data: bytes, pos: int
    ) -> Tuple[Optional[int], int]:
        """Read EBML vint from bytes. Returns (value, byte_length)."""
        if pos >= len(data):
            return (None, 0)

        first_byte = data[pos]
        if first_byte == 0:
            return (None, 0)

        length = 1
        mask = 0x80
        while not (first_byte & mask) and length < 8:
            length += 1
            mask >>= 1

        if pos + length > len(data):
            return (None, 0)

        value = first_byte & (mask - 1)
        for i in range(1, length):
            value = (value << 8) | data[pos + i]

        return (value, length)

    def _read_uint_from_bytes(self, data: bytes, pos: int, length: int) -> int:
        """Read unsigned integer from bytes."""
        value = 0
        for i in range(length):
            if pos + i < len(data):
                value = (value << 8) | data[pos + i]
        return value


__all__ = ["MKVHandler"]

