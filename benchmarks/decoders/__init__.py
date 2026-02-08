"""Decoder implementations for benchmarks.

The registry maps decoder names to either *classes* (instantiated with no
args) or *pre-built instances* (used as-is).  This allows parameterised
decoders like TorchCodec to expose multiple configurations.
"""

from __future__ import annotations

from typing import Union

from benchmarks.decoders.avdec_decoder import AvdecDecoder
from benchmarks.protocol import VideoDecoderProtocol

# Values are either a class (called with no args) or a ready instance.
_Entry = Union[type[VideoDecoderProtocol], VideoDecoderProtocol]

DECODERS: dict[str, _Entry] = {
    "avdec": AvdecDecoder,
}

# Optional decoders — register only when the library is importable.
try:
    from benchmarks.decoders.torchcodec_decoder import TORCHCODEC_CONFIGS

    for _cfg in TORCHCODEC_CONFIGS:
        DECODERS[_cfg.name] = _cfg
except ImportError:
    pass

try:
    import torch as _torch

    if _torch.cuda.is_available():
        from benchmarks.decoders.torchcodec_decoder import TORCHCODEC_GPU_CONFIGS

        for _cfg in TORCHCODEC_GPU_CONFIGS:
            DECODERS[_cfg.name] = _cfg
except ImportError:
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
    """Return a decoder by name (instantiating if needed)."""
    if name not in DECODERS:
        available = ", ".join(DECODERS.keys())
        raise ValueError(f"Unknown decoder: {name}. Available: {available}")
    entry = DECODERS[name]
    if isinstance(entry, type):
        return entry()
    return entry  # already an instance


def list_available_decoders() -> list[str]:
    """Return names of all importable decoders."""
    return list(DECODERS.keys())
