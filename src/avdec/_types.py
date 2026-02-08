"""Type definitions for avdec.

Mirrors the type definitions from MediaRef's video_decoder module.
"""

import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# FrameBatch
# ---------------------------------------------------------------------------


# Copied from https://github.com/pytorch/torchcodec/blob/main/src/torchcodec/_frame.py#L14-L27
def _frame_repr(self):
    """Print shape of ``.data`` instead of the (potentially huge) array."""
    s = self.__class__.__name__ + ":\n"
    spaces = "  "
    for field in dataclasses.fields(self):
        field_name = field.name
        field_val = getattr(self, field_name)
        if field_name == "data":
            field_name = "data (shape)"
            field_val = field_val.shape
        s += f"{spaces}{field_name}: {field_val}\n"
    return s


@dataclass
class FrameBatch:
    """Batch of video frames with timing information in NCHW format.

    Attributes:
        data: Frame data in NCHW format ``(N, C, H, W)`` with ``uint8`` dtype.
        pts_seconds: Presentation timestamps in seconds for each frame ``(N,)``.
        duration_seconds: Duration of each frame in seconds ``(N,)``.
    """

    data: npt.NDArray[np.uint8]  # [N, C, H, W]
    pts_seconds: npt.NDArray[np.float64]  # [N]
    duration_seconds: npt.NDArray[np.float64]  # [N]

    __repr__ = _frame_repr


# ---------------------------------------------------------------------------
# VideoStreamMetadata
# ---------------------------------------------------------------------------


@dataclass
class VideoStreamMetadata:
    """Video stream metadata.

    Attributes:
        num_frames: Total number of frames in the video.
        duration_seconds: Video duration in seconds (as ``Fraction`` for precision).
        average_rate: Average frame rate (as ``Fraction`` for precision).
        width: Frame width in pixels.
        height: Frame height in pixels.
        begin_stream_seconds: First frame PTS in seconds (default ``0``).
        end_stream_seconds: End of stream in seconds
            (``last_frame.pts + last_frame.duration``).
    """

    num_frames: int
    duration_seconds: Fraction
    average_rate: Fraction
    width: int
    height: int
    begin_stream_seconds: Fraction = Fraction(0)
    end_stream_seconds: Optional[Fraction] = None

    def __post_init__(self):
        """Set *end_stream_seconds* to ``begin + duration`` if not provided."""
        if self.end_stream_seconds is None:
            self.end_stream_seconds = self.begin_stream_seconds + self.duration_seconds


__all__ = [
    "FrameBatch",
    "VideoStreamMetadata",
]
