"""Public metadata names follow TorchCodec 0.17.0."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass
class StreamMetadata:
    duration_seconds_from_header: float | None
    begin_stream_seconds_from_header: float | None
    bit_rate: float | None
    codec: str | None
    stream_index: int
    media_type: str


@dataclass
class VideoStreamMetadata(StreamMetadata):
    width: int | None
    height: int | None
    num_frames_from_header: int | None
    average_fps_from_header: float | None
    pixel_aspect_ratio: Fraction | None
    rotation: float | None
    color_primaries: str | None
    color_space: str | None
    color_transfer_characteristic: str | None
    pixel_format: str | None
    begin_stream_seconds_from_content: float | None
    end_stream_seconds_from_content: float | None
    num_frames_from_content: int | None
    duration_seconds: float | None
    begin_stream_seconds: float | None
    end_stream_seconds: float | None
    num_frames: int | None
    average_fps: float | None


@dataclass
class AudioStreamMetadata(StreamMetadata):
    sample_rate: int | None
    num_channels: int | None
    sample_format: str | None
    duration_seconds: float | None
    begin_stream_seconds: float | None
