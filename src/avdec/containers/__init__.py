"""Container-specific optimizations for avdec.

This module provides a plugin architecture for container-specific optimizations.
Each container format (MP4, MKV, etc.) can implement optimized:
- Frame index building (fast metadata reading without packet scan)
- Seek operations (container-specific seek hints)

The system automatically detects container format and applies optimizations
when available, falling back to generic PyAV-based methods otherwise.
"""

from avdec.containers.base import ContainerHandler, get_handler, register_handler
from avdec.containers.mp4 import MP4Handler
from avdec.containers.mkv import MKVHandler

# Register built-in handlers
register_handler(MP4Handler())
register_handler(MKVHandler())

__all__ = [
    "ContainerHandler",
    "get_handler",
    "register_handler",
    "MP4Handler",
    "MKVHandler",
]

