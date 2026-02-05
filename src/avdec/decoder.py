"""TorchCodec-compatible video decoder using PyAV.

# =============================================================================
# TorchCodec Compatibility Notes
# =============================================================================
# This module implements a TorchCodec-compatible video decoder.
# The following interfaces match torchcodec.decoders.VideoDecoder:
#
#   - VideoDecoder class              -> torchcodec.decoders.VideoDecoder
#   - VideoDecoder.get_frame_at()     -> VideoDecoder.get_frame_at()
#   - VideoDecoder.get_frames_at()    -> VideoDecoder.get_frames_at()
#   - VideoDecoder.get_frame_played_at() -> VideoDecoder.get_frame_played_at()
#   - VideoDecoder.get_frames_played_at() -> VideoDecoder.get_frames_played_at()
#   - VideoDecoder.get_frames_in_range() -> VideoDecoder.get_frames_in_range()
#   - VideoDecoder.get_frames_played_in_range() -> VideoDecoder.get_frames_played_in_range()
#   - VideoDecoder.metadata           -> VideoDecoder.metadata
#   - VideoDecoder[idx]               -> VideoDecoder.__getitem__()
#   - len(VideoDecoder)               -> VideoDecoder.__len__()
#   - seek_mode parameter             -> VideoDecoder(seek_mode=...)
#   - dimension_order parameter       -> VideoDecoder(dimension_order=...)
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
from avdec._types import (
    DimensionOrder,
    Frame,
    FrameBatch,
    FrameIndex,
    SeekMode,
    VideoStreamMetadata,
)

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
            using FFmpeg's av_find_best_stream()
        dimension_order: Output dimension order, "NCHW" (default) or "NHWC"
            [TorchCodec Compatibility: VideoDecoder(dimension_order=...)]

    Example:
        >>> decoder = VideoDecoder("video.mp4")
        >>> frames = decoder.get_frames_at([0, 10, 20])
        >>> print(frames.data.shape)  # (3, 3, H, W) for NCHW
        >>> decoder.close()
    """

    def __init__(
        self,
        source: PathLike,
        *,
        seek_mode: SeekMode = SeekMode.EXACT,
        stream_index: Optional[int] = None,
        dimension_order: DimensionOrder = "NCHW",
    ):
        # Validate dimension_order
        if dimension_order not in ("NCHW", "NHWC"):
            raise ValueError(
                f"Invalid dimension_order ({dimension_order}). "
                "Supported values are 'NCHW', 'NHWC'."
            )

        self._source = Path(source) if isinstance(source, str) else source
        self._seek_mode = seek_mode
        self._dimension_order = dimension_order
        self._container: av.InputContainer = av.open(str(self._source), "r")
        self._stream = self._select_video_stream(stream_index)
        self._stream_index = self._stream.index

        # Build frame index for exact mode (before metadata extraction)
        if seek_mode == SeekMode.EXACT:
            self._frame_index: Optional[FrameIndex] = get_frame_index(
                self._source, self._container, self._stream_index
            )
        else:
            self._frame_index = None

        # Extract metadata (uses frame_index if available)
        self._metadata = self._extract_metadata()

        # Cache begin/end stream seconds for boundary validation
        self._begin_stream_seconds = self._metadata.begin_stream_seconds
        self._end_stream_seconds = (
            self._metadata.end_stream_seconds
            if self._metadata.end_stream_seconds is not None
            else self._metadata.duration_seconds
        )

    def _select_video_stream(self, stream_index: Optional[int]) -> av.VideoStream:
        """Select video stream by index or find the best one.

        [TorchCodec Compatibility: Best stream selection]
        If stream_index is None, uses FFmpeg's av_find_best_stream() via PyAV.
        This matches TorchCodec's SingleStreamDecoder::getBestStreamIndex().
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

        # Use FFmpeg's av_find_best_stream() via PyAV
        best_stream = self._container.streams.best('video')
        if best_stream is None:
            raise ValueError("No best video stream found")
        return best_stream

    def _extract_metadata(self) -> VideoStreamMetadata:
        """Extract video stream metadata from container.

        [TorchCodec Compatibility: VideoStreamMetadata fields]
        Populates all TorchCodec metadata fields with appropriate fallback logic.
        """
        stream = self._stream
        container = self._container

        # Duration from header
        duration_seconds_from_header: Optional[float] = None
        if stream.duration and stream.time_base:
            duration_seconds_from_header = float(stream.duration * stream.time_base)
        elif container.duration:
            duration_seconds_from_header = container.duration / av.time_base

        # FPS from header
        average_fps_from_header: Optional[float] = None
        if stream.average_rate:
            average_fps_from_header = float(stream.average_rate)

        # Frame count from header
        num_frames_from_header: Optional[int] = stream.frames if stream.frames else None

        # Compute values from content (frame index) if available
        num_frames_from_content: Optional[int] = None
        begin_stream_seconds_from_content: Optional[float] = None
        end_stream_seconds_from_content: Optional[float] = None

        if self._frame_index and self._frame_index.frame_infos:
            num_frames_from_content = len(self._frame_index.frame_infos)
            begin_stream_seconds_from_content = self._frame_index.frame_infos[0].pts
            last_frame = self._frame_index.frame_infos[-1]
            if last_frame.next_pts != float("inf"):
                end_stream_seconds_from_content = last_frame.next_pts
            else:
                # Estimate last frame duration from average fps
                if average_fps_from_header:
                    end_stream_seconds_from_content = (
                        last_frame.pts + 1.0 / average_fps_from_header
                    )
                else:
                    end_stream_seconds_from_content = last_frame.pts

        # Compute final values with fallback logic (matching TorchCodec)
        # num_frames: prefer content, fallback to header, fallback to duration*fps
        if num_frames_from_content is not None:
            num_frames = num_frames_from_content
        elif num_frames_from_header is not None:
            num_frames = num_frames_from_header
        elif duration_seconds_from_header and average_fps_from_header:
            num_frames = int(duration_seconds_from_header * average_fps_from_header)
        else:
            raise ValueError("Failed to determine frame count")

        # duration_seconds: prefer content-based, fallback to header
        if (
            begin_stream_seconds_from_content is not None
            and end_stream_seconds_from_content is not None
        ):
            duration_seconds = (
                end_stream_seconds_from_content - begin_stream_seconds_from_content
            )
        elif duration_seconds_from_header is not None:
            duration_seconds = duration_seconds_from_header
        else:
            raise ValueError("Failed to determine video duration")

        # average_fps: prefer content-based, fallback to header
        if num_frames_from_content is not None and duration_seconds > 0:
            average_fps = num_frames_from_content / duration_seconds
        elif average_fps_from_header is not None:
            average_fps = average_fps_from_header
        else:
            raise ValueError("Failed to determine average frame rate")

        # begin_stream_seconds: prefer content, fallback to 0
        begin_stream_seconds = (
            begin_stream_seconds_from_content
            if begin_stream_seconds_from_content is not None
            else 0.0
        )

        # end_stream_seconds: prefer content, fallback to duration
        end_stream_seconds = (
            end_stream_seconds_from_content
            if end_stream_seconds_from_content is not None
            else duration_seconds
        )

        return VideoStreamMetadata(
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            average_fps=average_fps,
            width=stream.width,
            height=stream.height,
            codec=stream.codec_context.name if stream.codec_context else None,
            time_base=stream.time_base,
            # TorchCodec extended fields
            begin_stream_seconds=begin_stream_seconds,
            end_stream_seconds=end_stream_seconds,
            num_frames_from_header=num_frames_from_header,
            num_frames_from_content=num_frames_from_content,
            average_fps_from_header=average_fps_from_header,
            duration_seconds_from_header=duration_seconds_from_header,
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

    def _empty_frame_batch(self) -> FrameBatch:
        """Return an empty FrameBatch with correct dimension order."""
        h, w = self._metadata.height, self._metadata.width
        if self._dimension_order == "NCHW":
            shape = (0, 3, h, w)
        else:
            shape = (0, h, w, 3)
        return FrameBatch(
            data=np.empty(shape, dtype=np.uint8),
            pts_seconds=np.array([], dtype=np.float64),
            duration_seconds=np.array([], dtype=np.float64),
            frame_indices=np.array([], dtype=np.int64),
        )

    def get_frame_at(self, index: int) -> Frame:
        """Return a single frame at the given index.

        [TorchCodec Compatibility: VideoDecoder.get_frame_at()]

        Args:
            index: The index of the frame to retrieve.

        Returns:
            Frame: The frame at the given index.
        """
        batch = self.get_frames_at([index])
        return Frame(
            data=batch.data[0],
            pts_seconds=float(batch.pts_seconds[0]),
            duration_seconds=float(batch.duration_seconds[0]),
        )

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
            return self._empty_frame_batch()

        # Normalize negative indices using Python slice semantics
        num_frames = len(self)
        normalized_indices = []
        for idx in indices:
            if idx < 0:
                idx = num_frames + idx
            if idx < 0 or idx >= num_frames:
                raise IndexError(f"Frame index {idx} out of range [0, {num_frames})")
            normalized_indices.append(idx)

        # Convert indices to timestamps using frame index
        if self._frame_index is not None:
            seconds = [self._frame_index.frame_infos[idx].pts for idx in normalized_indices]
        else:
            # Approximate mode
            seconds = [idx / self._metadata.average_fps for idx in normalized_indices]

        return self._decode_frames(normalized_indices, seconds)

    def get_frame_played_at(self, seconds: float) -> Frame:
        """Return a single frame played at the given timestamp in seconds.

        [TorchCodec Compatibility: VideoDecoder.get_frame_played_at()]

        Args:
            seconds: The timestamp in seconds when the frame is played.

        Returns:
            Frame: The frame that is played at the given timestamp.

        Raises:
            IndexError: If timestamp is outside valid range.
        """
        # TorchCodec boundary validation
        if not self._begin_stream_seconds <= seconds < self._end_stream_seconds:
            raise IndexError(
                f"Invalid pts in seconds: {seconds}. "
                f"It must be greater than or equal to {self._begin_stream_seconds} "
                f"and less than {self._end_stream_seconds}."
            )

        batch = self.get_frames_played_at([seconds])
        return Frame(
            data=batch.data[0],
            pts_seconds=float(batch.pts_seconds[0]),
            duration_seconds=float(batch.duration_seconds[0]),
        )

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
            IndexError: If any timestamp is outside valid range
        """
        if not seconds:
            return self._empty_frame_batch()

        # TorchCodec boundary validation for each timestamp
        for s in seconds:
            if not self._begin_stream_seconds <= s < self._end_stream_seconds:
                raise IndexError(
                    f"Invalid pts in seconds: {s}. "
                    f"It must be greater than or equal to {self._begin_stream_seconds} "
                    f"and less than {self._end_stream_seconds}."
                )

        # Convert timestamps to frame indices
        indices = [self._seconds_to_index(s) for s in seconds]

        return self._decode_frames(indices, seconds)

    def get_frames_in_range(
        self, start: int, stop: int, step: int = 1
    ) -> FrameBatch:
        """Return multiple frames at the given index range.

        [TorchCodec Compatibility: VideoDecoder.get_frames_in_range()]
        Frames are in [start, stop).

        Args:
            start: Index of the first frame to retrieve.
            stop: End of indexing range (exclusive).
            step: Step size between frames. Default: 1.

        Returns:
            FrameBatch: The frames within the specified range.
        """
        # Use Python slice semantics for negative indices
        num_frames = len(self)
        start, stop, step = slice(start, stop, step).indices(num_frames)
        indices = list(range(start, stop, step))
        return self.get_frames_at(indices)

    def get_frames_played_in_range(
        self,
        start_seconds: float,
        stop_seconds: float,
    ) -> FrameBatch:
        """Returns multiple frames in the given time range.

        [TorchCodec Compatibility: VideoDecoder.get_frames_played_in_range()]
        Frames are in the half open range [start_seconds, stop_seconds).

        Args:
            start_seconds: Time, in seconds, of the start of the range.
            stop_seconds: Time, in seconds, of the end of the range (exclusive).

        Returns:
            FrameBatch: The frames within the specified range.

        Raises:
            ValueError: If start_seconds > stop_seconds or range is invalid.
        """
        if start_seconds > stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be less than or equal to stop seconds ({stop_seconds})."
            )
        if not self._begin_stream_seconds <= start_seconds < self._end_stream_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be greater than or equal to {self._begin_stream_seconds} "
                f"and less than {self._end_stream_seconds}."
            )
        if stop_seconds > self._end_stream_seconds:
            raise ValueError(
                f"Invalid stop seconds: {stop_seconds}. "
                f"It must be less than or equal to {self._end_stream_seconds}."
            )

        # Find all frames in range using frame index
        if self._frame_index is not None:
            indices = []
            seconds = []
            for i, info in enumerate(self._frame_index.frame_infos):
                if start_seconds <= info.pts < stop_seconds:
                    indices.append(i)
                    seconds.append(info.pts)
            if not indices:
                return self._empty_frame_batch()
            return self._decode_frames(indices, seconds)
        else:
            # Approximate mode: use FPS to estimate frames
            fps = self._metadata.average_fps
            start_idx = int(start_seconds * fps)
            stop_idx = int(stop_seconds * fps)
            indices = list(range(start_idx, stop_idx))
            if not indices:
                return self._empty_frame_batch()
            seconds = [idx / fps for idx in indices]
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

        # Stack frames and apply dimension order
        data = np.stack(frames_data, axis=0)  # NHWC format

        # Convert to NCHW if requested
        if self._dimension_order == "NCHW":
            data = np.transpose(data, (0, 3, 1, 2))  # NHWC -> NCHW

        return FrameBatch(
            data=data,
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

