"""Video decoder using PyAV with playback-frame semantics.

Mirrors the interface from MediaRef's ``PyAVVideoDecoder``.

Core API:
    - ``get_frames_played_at(seconds)``  — frames at specific timestamps
    - ``get_frames_played_in_range(start, stop, fps=None)`` — frames in a time window

Example:
    >>> with VideoDecoder("video.mp4") as decoder:
    ...     batch = decoder.get_frames_played_at([0.0, 0.5, 1.0])
    ...     print(batch.data.shape)  # (3, 3, H, W) NCHW
"""

from __future__ import annotations

import gc
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Union

import av
import numpy as np

from avdec._types import FrameBatch, VideoStreamMetadata

PathLike = Union[str, Path]

# Garbage collection interval for PyAV reference cycles.
# Reference: https://github.com/pytorch/vision/blob/428a54c96e82226c0d2d8522e9cbfdca64283da0/torchvision/io/video.py#L53-L55
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10


def _convert_av_frames_to_nchw(av_frames: List[av.VideoFrame]) -> List[np.ndarray]:
    """Convert a list of PyAV frames to NCHW numpy arrays (RGB)."""
    frames = []
    for frame in av_frames:
        rgb = frame.to_ndarray(format="rgb24")  # HWC
        frames.append(np.transpose(rgb, (2, 0, 1)))  # CHW
    return frames


class VideoDecoder:
    """Video decoder for ML training workloads.

    Uses playback-frame semantics: returns frame *i* where
    ``frame[i].pts <= timestamp < frame[i+1].pts``.

    Args:
        source: Path to video file or URL.

    Example:
        >>> with VideoDecoder("video.mp4") as decoder:
        ...     batch = decoder.get_frames_played_at([0.0, 0.5, 1.0])
        ...     print(batch.data.shape)  # (3, 3, H, W) NCHW
    """

    def __init__(self, source: PathLike):
        self._source = source
        self._container: av.InputContainer = av.open(str(source), "r")
        self._metadata = self._extract_metadata()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _extract_metadata(self) -> VideoStreamMetadata:
        """Extract video stream metadata from container.

        Decodes only the first frame to get accurate ``begin_stream_seconds``.
        Uses header metadata for duration / end_stream_seconds.
        """
        container = self._container
        if not container.streams.video:
            raise ValueError(f"No video streams found in {self._source}")
        stream = container.streams.video[0]

        # Frame rate
        if stream.average_rate:
            average_rate = Fraction(stream.average_rate)
        else:
            raise ValueError("Failed to determine average rate")

        # Duration from header
        if stream.duration and stream.time_base:
            duration_seconds = Fraction(stream.duration * stream.time_base)
        elif container.duration:
            duration_seconds = Fraction(container.duration, av.time_base)
        else:
            raise ValueError("Failed to determine duration")

        # Decode first frame to get accurate begin_stream_seconds
        container.seek(0)
        first_pts = Fraction(0)
        for frame in container.decode(video=0):
            if frame.time is not None:
                first_pts = Fraction(frame.time).limit_denominator(1_000_000)
            break
        container.seek(0)

        begin_stream_seconds = first_pts
        end_stream_seconds = begin_stream_seconds + duration_seconds

        num_frames = stream.frames if stream.frames else int(duration_seconds * average_rate)

        return VideoStreamMetadata(
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            average_rate=average_rate,
            width=stream.width,
            height=stream.height,
            begin_stream_seconds=begin_stream_seconds,
            end_stream_seconds=end_stream_seconds,
        )

    @property
    def metadata(self) -> VideoStreamMetadata:
        """Access video stream metadata."""
        return self._metadata

    def _create_empty_batch(self) -> FrameBatch:
        """Create an empty FrameBatch with correct spatial dimensions."""
        return FrameBatch(
            data=np.empty((0, 3, self._metadata.height, self._metadata.width), dtype=np.uint8),
            pts_seconds=np.array([], dtype=np.float64),
            duration_seconds=np.array([], dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frames_played_at(self, seconds: List[float]) -> FrameBatch:
        """Retrieve frames that would be displayed at specific timestamps.

        Uses playback-frame semantics: returns frame *i* where
        ``frame[i].pts <= timestamp < frame[i+1].pts``.

        Args:
            seconds: List of timestamps in seconds.

        Returns:
            FrameBatch with frame data in NCHW format.

        Raises:
            ValueError: If any timestamp is outside
                ``[begin_stream_seconds, end_stream_seconds)``.
        """
        if not seconds:
            return self._create_empty_batch()

        # Validate timestamps
        begin_stream = float(self._metadata.begin_stream_seconds)
        end_stream = float(self._metadata.end_stream_seconds)  # type: ignore[arg-type]
        for t in seconds:
            if t < begin_stream:
                raise ValueError(f"Timestamp {t}s < begin_stream_seconds ({begin_stream}s)")
            if t >= end_stream:
                raise ValueError(f"Timestamp {t}s >= end_stream_seconds ({end_stream}s)")

        # Get frames using playback semantics
        av_frames = self._get_frames_played_at(seconds)

        # Convert to RGB numpy arrays in NCHW format
        frames = _convert_av_frames_to_nchw(av_frames)

        pts_list = [float(frame.time) for frame in av_frames]
        duration = float(1.0 / self._metadata.average_rate)

        return FrameBatch(
            data=np.stack(frames, axis=0),
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.full(len(seconds), duration, dtype=np.float64),
        )

    def get_frames_played_in_range(
        self,
        start_seconds: float,
        stop_seconds: float,
        fps: Optional[float] = None,
    ) -> FrameBatch:
        """Return frames in the half-open range ``[start_seconds, stop_seconds)``.

        Args:
            start_seconds: Start of the range (inclusive), in seconds.
            stop_seconds: End of the range (exclusive), in seconds.
            fps: If specified, resample output to this frame rate by
                duplicating/dropping frames. If ``None``, returns frames
                at the source video's native rate.

        Returns:
            FrameBatch with frame data in NCHW format.

        Raises:
            ValueError: If the range parameters are invalid.
        """
        begin_stream = float(self._metadata.begin_stream_seconds)
        end_stream = float(self._metadata.end_stream_seconds)  # type: ignore[arg-type]

        if not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be less than or equal to stop seconds ({stop_seconds})."
            )
        if not begin_stream <= start_seconds < end_stream:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be greater than or equal to {begin_stream} "
                f"and less than {end_stream}."
            )
        if not stop_seconds <= end_stream:
            raise ValueError(f"Invalid stop seconds: {stop_seconds}. It must be less than or equal to {end_stream}.")

        # Resampled mode: generate timestamps at the given fps
        if fps is not None:
            timestamps = np.arange(start_seconds, stop_seconds, 1.0 / fps).tolist()
            if not timestamps:
                return self._create_empty_batch()
            return self.get_frames_played_at(timestamps)

        # Native frame rate: decode all frames with pts in [start, stop)
        self._seek_to_or_before(start_seconds)

        av_frames: list[av.VideoFrame] = []
        for frame in self._container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")
            frame_pts = float(frame.time)
            if frame_pts >= stop_seconds:
                break
            if frame_pts >= start_seconds:
                av_frames.append(frame)

        if not av_frames:
            return self._create_empty_batch()

        frames = _convert_av_frames_to_nchw(av_frames)

        pts_list = [float(frame.time) for frame in av_frames]
        duration = float(1.0 / self._metadata.average_rate)

        return FrameBatch(
            data=np.stack(frames, axis=0),
            pts_seconds=np.array(pts_list, dtype=np.float64),
            duration_seconds=np.full(len(av_frames), duration, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_frames_played_at(self, seconds: List[float]) -> list[av.VideoFrame]:
        """Get frames using playback-frame semantics (internal).

        For each timestamp returns the frame where
        ``frame[i].pts <= timestamp < frame[i+1].pts``.
        """
        global _CALLED_TIMES
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == 0:
            gc.collect()

        # Sort queries for efficient sequential access, preserving output order
        indexed_queries = sorted(enumerate(seconds), key=lambda x: x[1])
        results: list[av.VideoFrame] = [None] * len(seconds)  # type: ignore[list-item]

        query_idx = 0
        prev_frame: Optional[av.VideoFrame] = None
        prev_frame_pts: float = float("-inf")

        # Seek to just before the first query
        first_query_time = indexed_queries[0][1]
        self._seek_to_or_before(first_query_time)

        # Decode frames and match to queries using nextPts logic
        for frame in self._container.decode(video=0):
            if frame.time is None:
                raise ValueError("Frame time is None")

            frame_pts = float(frame.time)

            # Process all queries where prev_frame.pts <= query < frame.pts
            while query_idx < len(indexed_queries):
                orig_idx, query_time = indexed_queries[query_idx]
                if query_time < frame_pts:
                    if prev_frame is not None and prev_frame_pts <= query_time:
                        results[orig_idx] = prev_frame
                        query_idx += 1
                    elif prev_frame is None:
                        raise ValueError(f"Timestamp {query_time}s is before first frame (pts={frame_pts}s)")
                    else:
                        raise ValueError(f"Internal error: query {query_time}s not in [{prev_frame_pts}, {frame_pts})")
                else:
                    break

            prev_frame = frame
            prev_frame_pts = frame_pts

            if query_idx >= len(indexed_queries):
                break

        # Handle remaining queries (at or after last decoded frame)
        while query_idx < len(indexed_queries):
            orig_idx, query_time = indexed_queries[query_idx]
            if prev_frame is not None and prev_frame_pts <= query_time:
                results[orig_idx] = prev_frame
                query_idx += 1
            else:
                raise ValueError(f"Could not find frame for timestamp {query_time}s")

        for i, result in enumerate(results):
            if result is None:
                raise ValueError(f"Could not find frame for timestamp {seconds[i]}s")

        return results

    def _seek_to_or_before(self, target_seconds: float) -> None:
        """Seek to *target_seconds* or before it (exponential backoff).

        PyAV seeks to keyframes which may land past the target when
        keyframes are sparse.  This method detects overshooting and
        backs off exponentially until a valid position is found.
        """
        stream = self._container.streams.video[0]
        time_base = float(stream.time_base)
        begin_stream = float(self._metadata.begin_stream_seconds)

        seek_target = target_seconds
        buffer = 1.0  # initial backoff in seconds

        while True:
            seek_pts = int(seek_target / time_base)
            self._container.seek(
                seek_pts,
                stream=stream,
                any_frame=False,
                backward=True,
            )

            # Peek at the first decoded frame
            try:
                frame = next(self._container.decode(video=0))
            except StopIteration:
                return

            if frame.time is not None and frame.time <= target_seconds:
                # Landed at or before target — re-seek to restore position
                self._container.seek(
                    seek_pts,
                    stream=stream,
                    any_frame=False,
                    backward=True,
                )
                return

            # Overshot — back off
            seek_target = target_seconds - buffer
            buffer *= 2

            if seek_target <= begin_stream:
                self._container.seek(0)
                return

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release video decoder resources."""
        if hasattr(self, "_container"):
            self._container.close()

    def __enter__(self) -> "VideoDecoder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


__all__ = ["VideoDecoder"]
