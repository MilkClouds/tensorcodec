"""avdec - Batch video frame loader for ML training.

Drop-in TorchCodec API in pure Python — NumPy output, no PyTorch required.

Example:
    >>> from avdec import VideoDecoder
    >>> decoder = VideoDecoder("video.mp4")
    >>> frames = decoder.get_frames_at([0, 10, 20])
    >>> print(frames.data.shape)  # (3, 3, H, W) for NCHW (default)
    >>> decoder.close()

Diagnose videos for ML training:
    >>> import avdec
    >>> report = avdec.doctor("video.mp4")
    >>> print(report)
"""

from avdec._types import Frame, FrameBatch, FrameInfo, SeekMode, VideoStreamMetadata
from avdec.decoder import VideoDecoder
from avdec.doctor import doctor

try:
    from avdec._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "VideoDecoder",
    "VideoStreamMetadata",
    "Frame",
    "FrameBatch",
    "FrameInfo",
    "SeekMode",
    "doctor",
    "__version__",
]

