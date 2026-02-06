"""Decoder implementations for benchmarks."""

from __future__ import annotations

from benchmarks.decoders.avdec_decoder import AvdecDecoder
from benchmarks.protocol import VideoDecoderProtocol

# Registry of available decoders (avdec is always available).
DECODERS: dict[str, type[VideoDecoderProtocol]] = {
    "avdec": AvdecDecoder,
}

# Optional decoders — register only when the library is importable.
try:
    from benchmarks.decoders.torchcodec_decoder import TorchCodecDecoder
    DECODERS["torchcodec"] = TorchCodecDecoder
except (ImportError, RuntimeError):
    pass

try:
    from benchmarks.decoders.decord_decoder import DecordDecoder
    DECODERS["decord"] = DecordDecoder
except ImportError:
    pass

try:
    from benchmarks.decoders.opencv_decoder import OpenCVDecoder
    DECODERS["opencv"] = OpenCVDecoder
except ImportError:
    pass

try:
    from benchmarks.decoders.torchvision_decoder import (
        TorchVisionPyAVDecoder,
        TorchVisionVideoReaderDecoder,
        available_backends,
    )
    _tv_backends = available_backends()
    if "pyav" in _tv_backends:
        DECODERS["torchvision-pyav"] = TorchVisionPyAVDecoder
    if "video_reader" in _tv_backends:
        DECODERS["torchvision-video_reader"] = TorchVisionVideoReaderDecoder
except ImportError:
    pass


def get_decoder(name: str) -> VideoDecoderProtocol:
    """Instantiate a decoder by name."""
    if name not in DECODERS:
        available = ", ".join(DECODERS.keys())
        raise ValueError(f"Unknown decoder: {name}. Available: {available}")
    return DECODERS[name]()


def list_available_decoders() -> list[str]:
    """Return names of all importable decoders."""
    return list(DECODERS.keys())

