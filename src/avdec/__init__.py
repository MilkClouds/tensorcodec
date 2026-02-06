"""avdec — Video frame loader for ML training.

Pure-Python decoder using PyAV.  NumPy output, no PyTorch required.

Example:
    >>> from avdec import VideoDecoder
    >>> with VideoDecoder("video.mp4") as decoder:
    ...     batch = decoder.get_frames_played_at([0.0, 0.5, 1.0])
    ...     print(batch.data.shape)  # (3, 3, H, W) for NCHW
"""

from avdec._types import FrameBatch, VideoStreamMetadata
from avdec.decoder import VideoDecoder
from avdec.doctor import doctor

try:
    from avdec._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "VideoDecoder",
    "VideoStreamMetadata",
    "FrameBatch",
    "doctor",
    "__version__",
]

