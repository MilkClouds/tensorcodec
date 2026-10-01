"""Audio/video decoding into NumPy arrays; no torch or PyAV runtime dependency."""

from tensorcodec import decoders
from tensorcodec._frame import AudioSamples, Frame, FrameBatch

__version__ = "0.1.0"
__all__ = ["AudioSamples", "Frame", "FrameBatch", "decoders"]
