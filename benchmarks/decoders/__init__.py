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

