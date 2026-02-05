"""TorchCodec-compatible video decoder using PyAV.

# =============================================================================
# TorchCodec Compatibility Notes
# =============================================================================
# This module implements a TorchCodec-compatible video decoder.
# The following interfaces match torchcodec.decoders.VideoDecoder:
#
#   - VideoDecoder class              -> torchcodec.decoders.VideoDecoder
#   - VideoDecoder.get_frames_at()    -> VideoDecoder.get_frames_at()
#   - VideoDecoder.get_frames_played_at() -> VideoDecoder.get_frames_played_at()
#   - VideoDecoder.metadata           -> VideoDecoder.metadata
#   - VideoDecoder[idx]               -> VideoDecoder.__getitem__()
#   - len(VideoDecoder)               -> VideoDecoder.__len__()
#   - seek_mode parameter             -> VideoDecoder(seek_mode=...)
#
# DO NOT modify these interfaces without verifying TorchCodec compatibility.
# =============================================================================
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Dict, List, Optional, Union

import av
import numpy as np
import numpy.typing as npt

from avdec._metadata import get_frame_index
from avdec._types import FrameBatch, FrameIndex, SeekMode, VideoStreamMetadata

PathLike = Union[str, Path]


class VideoDecoder:
    """TorchCodec-compatible video decoder using PyAV.

    [TorchCodec Compatibility: torchcodec.decoders.VideoDecoder]

    This decoder implements TorchCodec's exact mode semantics:
    - Pre-scans all packets to build frame index
    - Uses playback frame semantics: frame[i].pts <= timestamp < frame[i+1].pts
    - Supports efficient batch frame loading

    Args:
        source: Path to video file
        seek_mode: SeekMode.EXACT (default) or SeekMode.APPROXIMATE
            [TorchCodec Compatibility: VideoDecoder(seek_mode=...)]
        stream_index: Video stream index. If None, selects the "best" stream
            (highest resolution, matching TorchCodec behavior)

    Example:
        >>> decoder = VideoDecoder("video.mp4")
        >>> frames = decoder.get_frames_at([0, 10, 20])
        >>> print(frames.data.shape)  # (3, H, W, 3)
        >>> decoder.close()
    """

    def __init__(
        self,
        source: PathLike,
        *,
        seek_mode: SeekMode = SeekMode.EXACT,
        stream_index: Optional[int] = None,
    ):
        self._source = Path(source) if isinstance(source, str) else source
        self._seek_mode = seek_mode
        self._container: av.InputContainer = av.open(str(self._source), "r")
        self._stream = self._select_video_stream(stream_index)
        self._stream_index = self._stream.index
        self._metadata = self._extract_metadata()

        # Build frame index for exact mode
        if seek_mode == SeekMode.EXACT:
            self._frame_index: Optional[FrameIndex] = get_frame_index(
                self._source, self._container, self._stream_index
            )
        else:
            self._frame_index = None

    def _select_video_stream(self, stream_index: Optional[int]) -> av.VideoStream:
        """Select video stream by index or find the best one.

        [TorchCodec Compatibility: Best stream selection]
        If stream_index is None, selects the "best" video stream based on:
        1. Highest resolution (width * height)
        2. If tie, uses FFmpeg's default ordering

        This matches TorchCodec's behavior of selecting the best video stream.
        TorchCodec's VideoCore::getBestVideoStreamIndex() uses av_find_best_stream().
        """
        video_streams = self._container.streams.video
        if not video_streams:
            raise ValueError("No video streams found in container")

        if stream_index is not None:
            # Find stream by absolute index in container
            for stream in self._container.streams:
                if stream.index == stream_index:
                    if stream.type != 'video':
                        raise ValueError(f"Stream {stream_index} is not a video stream")
                    return stream
            raise ValueError(f"Stream index {stream_index} not found in container")

        # Select best video stream (highest resolution)
        best_stream = video_streams[0]
        best_resolution = best_stream.width * best_stream.height

        for stream in video_streams[1:]:
            resolution = stream.width * stream.height
            if resolution > best_resolution:
                best_stream = stream
                best_resolution = resolution

        return best_stream

    def _extract_metadata(self) -> VideoStreamMetadata:
        """Extract video stream metadata from container."""
        stream = self._stream
        container = self._container

        # Determine duration
        if stream.duration and stream.time_base:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration:
            duration_seconds = container.duration / av.time_base
        else:
            raise ValueError("Failed to determine video duration")

        # Determine frame rate
        if stream.average_rate:
            average_fps = float(stream.average_rate)
        else:
            raise ValueError("Failed to determine average frame rate")

        # Determine frame count
        if stream.frames:
            num_frames = stream.frames
        else:
            num_frames = int(duration_seconds * average_fps)

        return VideoStreamMetadata(
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            average_fps=average_fps,
            width=stream.width,
            height=stream.height,
            codec=stream.codec_context.name if stream.codec_context else None,
            time_base=stream.time_base,
        )

    @property
    def metadata(self) -> VideoStreamMetadata:
        """Access video stream metadata.

        [TorchCodec Compatibility: VideoDecoder.metadata]
        Returns metadata matching TorchCodec's VideoStreamMetadata structure.
        """
        return self._metadata

    @property
    def source(self) -> Path:
        """Source file path."""
        return self._source

    def __len__(self) -> int:
        """Return number of frames.

        [TorchCodec Compatibility: len(VideoDecoder)]
        """
        if self._frame_index:
            return len(self._frame_index)
        return self._metadata.num_frames

    def _seconds_to_index(self, seconds: float) -> int:
        """Convert timestamp to frame index using TorchCodec semantics.

        [TorchCodec Compatibility: Playback frame semantics]
        Finds frame i where: frame[i].pts <= seconds < frame[i+1].pts

        This implements TorchCodec's getPtsSecondsForFrame() logic using
        bisect_right on the PTS list:
        - bisect_right returns the insertion point after any existing entries equal to seconds
        - Subtracting 1 gives us the frame whose PTS is <= seconds

        Reference: TorchCodec's VideoDecoder.cpp, getFrameAtIndexInternal()

        Args:
            seconds: Timestamp in seconds

        Returns:
            Frame index

        Raises:
            ValueError: If timestamp is before first frame or after video duration
        """
        if self._frame_index is None:
            # Approximate mode: use average FPS
            idx = int(seconds * self._metadata.average_fps)
            return max(0, min(idx, self._metadata.num_frames - 1))

        pts_list = self._frame_index.pts_list
        if not pts_list:
            raise ValueError("Empty frame index")

        # bisect_right returns insertion point after equal values
        # So bisect_right - 1 gives us the frame with pts <= seconds
        idx = bisect.bisect_right(pts_list, seconds) - 1

        if idx < 0:
            raise ValueError(
                f"Timestamp {seconds}s is before first frame (pts={pts_list[0]}s)"
            )

        # For timestamps beyond last frame, return last frame
        # (TorchCodec behavior: last frame extends to infinity)
        return min(idx, len(pts_list) - 1)

    def _find_keyframe_before(self, frame_index: int) -> int:
        """Find the keyframe at or before the given frame index."""
        if self._frame_index is None or not self._frame_index.keyframe_indices:
            return 0

        keyframes = self._frame_index.keyframe_indices
        # Find rightmost keyframe <= frame_index
        pos = bisect.bisect_right(keyframes, frame_index)
        if pos == 0:
            return keyframes[0]
        return keyframes[pos - 1]

    def get_frames_at(self, indices: List[int]) -> FrameBatch:
        """Retrieve frames at specific frame indices.

        [TorchCodec Compatibility: VideoDecoder.get_frames_at()]
        Returns frames at given indices, matching TorchCodec's exact mode behavior.

        Args:
            indices: List of frame indices to retrieve

        Returns:
            FrameBatch containing frame data and timing information
        """
        if not indices:
            return FrameBatch(
                data=np.empty((0, self._metadata.height, self._metadata.width, 3), dtype=np.uint8),
                pts_seconds=np.array([], dtype=np.float64),
                duration_seconds=np.array([], dtype=np.float64),
                frame_indices=np.array([], dtype=np.int64),
            )

        # Normalize negative indices
        num_frames = len(self)
        indices = [idx % num_frames for idx in indices]

        # Convert indices to timestamps using frame index
        if self._frame_index is not None:
            seconds = [self._frame_index.frame_infos[idx].pts for idx in indices]
        else:
            # Approximate mode
            seconds = [idx / self._metadata.average_fps for idx in indices]

        return self._decode_frames(indices, seconds)

    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames at specific timestamps.

        [TorchCodec Compatibility: VideoDecoder.get_frames_played_at()]
        Uses TorchCodec's playback frame semantics:
        Returns frame i where frame[i].pts <= timestamp < frame[i+1].pts

        This is the core timestamp-to-frame mapping that must match TorchCodec exactly.

        Args:
            seconds: List of timestamps in seconds

        Returns:
            FrameBatch containing frame data and timing information

        Raises:
            ValueError: If any timestamp exceeds video duration
        """
        if not seconds:
            return FrameBatch(
                data=np.empty((0, self._metadata.height, self._metadata.width, 3), dtype=np.uint8),
                pts_seconds=np.array([], dtype=np.float64),
                duration_seconds=np.array([], dtype=np.float64),
                frame_indices=np.array([], dtype=np.int64),
            )

        max_seconds = max(seconds)
        if max_seconds > self._metadata.duration_seconds:
            raise ValueError(
                f"Timestamp {max_seconds}s exceeds video duration {self._metadata.duration_seconds}s"
            )

        # Convert timestamps to frame indices
        indices = [self._seconds_to_index(s) for s in seconds]

        return self._decode_frames(indices, seconds)

    def _decode_frames(
        self, indices: List[int], query_seconds: List[float]
    ) -> FrameBatch:
        """Decode frames at given indices.

        Uses sequential decoding with efficient keyframe seeking.
        """
        # Build mapping: (original_position, frame_index, query_seconds)
        queries = [(i, idx, sec) for i, (idx, sec) in enumerate(zip(indices, query_seconds))]

        # Sort by frame index for efficient sequential decoding
        queries_sorted = sorted(queries, key=lambda x: x[1])

        # Decode frames
        decoded: Dict[int, av.VideoFrame] = {}
        unique_indices = sorted(set(idx for _, idx, _ in queries_sorted))

        if unique_indices:
            # Seek to keyframe before first requested frame
            first_keyframe_idx = self._find_keyframe_before(unique_indices[0])
            if self._frame_index and first_keyframe_idx < len(self._frame_index.frame_infos):
                seek_pts = self._frame_index.frame_infos[first_keyframe_idx].pts
            else:
                seek_pts = first_keyframe_idx / self._metadata.average_fps

            # Seek to position
            seek_ts = int(seek_pts / float(self._stream.time_base))
            self._container.seek(seek_ts, stream=self._stream)

            # Decode sequentially
            target_set = set(unique_indices)
            current_idx = first_keyframe_idx

            for frame in self._container.decode(self._stream):
                if current_idx in target_set:
                    decoded[current_idx] = frame
                    target_set.remove(current_idx)

                current_idx += 1

                if not target_set or current_idx > unique_indices[-1]:
                    break

        # Convert to numpy and reorder to original query order
        frames_data = []
        pts_list = []
        duration_list = []
        frame_indices_list = []

        for orig_pos, idx, query_sec in queries:
            if idx in decoded:
                frame = decoded[idx]
                # Convert to RGB numpy array
                rgb_frame = frame.to_ndarray(format="rgb24")
                frames_data.append(rgb_frame)
                pts_list.append(frame.time if frame.time else query_sec)
            else:
                # Frame not found - use placeholder
                h, w = self._metadata.height, self._metadata.width
                frames_data.append(np.zeros((h, w, 3), dtype=np.uint8))
                pts_list.append(query_sec)

            # Calculate duration from frame index
            if self._frame_index and idx < len(self._frame_index.frame_infos):
                info = self._frame_index.frame_infos[idx]
                if info.next_pts != float("inf"):
                    duration = info.next_pts - info.pts
                else:
                    duration = 1.0 / self._metadata.average_fps
            else:
                duration = 1.0 / self._metadata.average_fps

            duration_list.append(duration)
            frame_indices_list.append(idx)

        return FrameBatch(
            data=np.stack(frames_data, axis=0),
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.array(duration_list, dtype=np.float64),
            frame_indices=np.array(frame_indices_list, dtype=np.int64),
        )

    def __getitem__(self, key: Union[int, slice]) -> npt.NDArray[np.uint8]:
        """Enable array-like indexing for frame access.

        [TorchCodec Compatibility: VideoDecoder.__getitem__()]
        Supports decoder[idx] and decoder[start:stop:step] syntax.
        """
        if isinstance(key, int):
            return self.get_frames_at([key]).data[0]
        elif isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return self.get_frames_at(list(indices)).data
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def close(self) -> None:
        """Release video decoder resources."""
        if hasattr(self, "_container") and self._container:
            self._container.close()
            self._container = None  # type: ignore

    def __repr__(self) -> str:
        """Return string representation."""
        status = "closed" if self._container is None else "open"
        return (
            f"VideoDecoder({self._source.name!r}, "
            f"frames={self._metadata.num_frames}, "
            f"{self._metadata.width}x{self._metadata.height}, "
            f"status={status})"
        )

    def __enter__(self) -> "VideoDecoder":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure resources are released."""
        self.close()


__all__ = ["VideoDecoder"]

