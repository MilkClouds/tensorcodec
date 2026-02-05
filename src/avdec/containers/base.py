"""Base class and registry for container handlers.

Container handlers provide format-specific optimizations for:
1. Frame index building - Reading timing info from container metadata
2. Seek optimization - Container-specific seek strategies
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

from avdec._types import FrameIndex

if TYPE_CHECKING:
    import av

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Global registry of container handlers
_handlers: Dict[str, "ContainerHandler"] = {}


class ContainerHandler(ABC):
    """Base class for container-specific optimizations.
    
    Subclasses implement format-specific optimizations for frame index
    building and seek operations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name for logging/debugging."""
        ...

    @property
    @abstractmethod
    def extensions(self) -> Set[str]:
        """File extensions this handler supports (lowercase, with dot)."""
        ...

    @abstractmethod
    def can_handle(self, source: PathLike) -> bool:
        """Check if this handler can process the given file.
        
        May perform additional checks beyond extension matching
        (e.g., magic bytes verification).
        """
        ...

    @abstractmethod
    def build_frame_index(
        self,
        source: PathLike,
        stream_index: Optional[int] = None,
    ) -> Optional[FrameIndex]:
        """Build frame index from container metadata.
        
        Returns None if fast index building is not possible,
        triggering fallback to packet scan.
        
        Args:
            source: Path to video file
            stream_index: Video stream index (None = best/first)
            
        Returns:
            FrameIndex if successful, None to fall back to packet scan
        """
        ...

    def get_keyframe_positions(
        self,
        source: PathLike,
        stream_index: Optional[int] = None,
    ) -> Optional[List[int]]:
        """Get keyframe byte positions for optimized seeking.
        
        Returns None if not available, using default seek behavior.
        
        Args:
            source: Path to video file
            stream_index: Video stream index
            
        Returns:
            List of byte positions for keyframes, or None
        """
        return None


def register_handler(handler: ContainerHandler) -> None:
    """Register a container handler."""
    for ext in handler.extensions:
        _handlers[ext.lower()] = handler
    logger.debug(f"Registered container handler: {handler.name} for {handler.extensions}")


def get_handler(source: PathLike) -> Optional[ContainerHandler]:
    """Get the appropriate handler for a file.
    
    Args:
        source: Path to video file
        
    Returns:
        ContainerHandler if one matches, None otherwise
    """
    path = Path(source)
    ext = path.suffix.lower()
    
    handler = _handlers.get(ext)
    if handler and handler.can_handle(source):
        return handler
    
    return None


def get_frame_index_optimized(
    source: PathLike,
    container: "av.InputContainer",
    stream_index: Optional[int] = None,
) -> Optional[FrameIndex]:
    """Try to build frame index using container-specific optimization.
    
    Args:
        source: Path to video file
        container: Open PyAV container (for fallback info)
        stream_index: Video stream index
        
    Returns:
        FrameIndex if optimization succeeded, None to use fallback
    """
    handler = get_handler(source)
    if handler is None:
        return None
    
    try:
        frame_index = handler.build_frame_index(source, stream_index)
        if frame_index is not None:
            logger.debug(f"Used {handler.name} optimization for {source}")
            return frame_index
    except Exception as e:
        logger.debug(f"{handler.name} optimization failed for {source}: {e}")
    
    return None


__all__ = [
    "ContainerHandler",
    "register_handler",
    "get_handler",
    "get_frame_index_optimized",
]

