"""Diagnostic report and issue structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Severity(Enum):
    """Issue severity level."""
    OK = "ok"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass
class Issue:
    """A diagnosed issue with contextual explanation."""
    
    severity: Severity
    title: str
    body: str  # Multi-line formatted explanation
    fix_command: Optional[str] = None
    
    def __str__(self) -> str:
        icons = {
            Severity.OK: "✓",
            Severity.INFO: "ℹ",
            Severity.WARN: "⚠",
            Severity.ERROR: "✗",
        }
        result = f"{icons[self.severity]} {self.title}\n{self.body}"
        if self.fix_command:
            result += f"\n\n   $ {self.fix_command}"
        return result


@dataclass
class VideoStats:
    """Analyzed video statistics."""
    
    path: Path
    container: str
    codec: str
    width: int
    height: int
    duration_sec: float
    frame_count: int
    fps_avg: float
    file_size_bytes: int
    
    # Keyframe analysis
    keyframe_count: int = 0
    keyframe_interval_frames_avg: float = 0
    keyframe_interval_frames_max: int = 0
    keyframe_interval_sec_avg: float = 0
    
    # Frame timing
    is_cfr: bool = True  # Constant frame rate
    frame_duration_min_ms: float = 0
    frame_duration_max_ms: float = 0
    frame_duration_stddev_ms: float = 0
    
    # Container specifics
    moov_position: str = "unknown"  # "start", "end", "unknown"
    
    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)


@dataclass  
class DiagnosticReport:
    """Complete diagnostic report for a video file."""
    
    stats: Optional[VideoStats] = None
    issues: List[Issue] = field(default_factory=list)
    error: Optional[str] = None  # If analysis failed
    
    @property
    def ok(self) -> bool:
        """No warnings or errors."""
        return all(i.severity in (Severity.OK, Severity.INFO) for i in self.issues)
    
    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARN]
    
    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]
    
    def __str__(self) -> str:
        if self.error:
            return f"✗ Analysis failed: {self.error}"
        
        if not self.stats:
            return "✗ No video analyzed"
        
        lines = [self._header(), ""]
        
        if self.ok:
            lines.append("✓ No issues found. This video is well-suited for ML training.")
        else:
            for issue in self.issues:
                if issue.severity in (Severity.WARN, Severity.ERROR):
                    lines.append(str(issue))
                    lines.append("")
        
        # Info section at the end
        infos = [i for i in self.issues if i.severity == Severity.INFO]
        if infos:
            lines.append("── Additional Info ──")
            for issue in infos:
                lines.append(str(issue))
                lines.append("")
        
        return "\n".join(lines)
    
    def _header(self) -> str:
        s = self.stats
        return (
            f"┌ {s.path.name}\n"
            f"│ {s.container} / {s.codec} / {s.width}×{s.height}\n"
            f"│ {s.frame_count} frames / {s.duration_sec:.1f}s / {s.fps_avg:.2f}fps\n"
            f"└ {s.file_size_mb:.1f} MB"
        )

