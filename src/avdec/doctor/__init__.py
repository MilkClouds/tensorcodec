"""Video diagnostics for ML training.

Example:
    >>> import avdec
    >>> report = avdec.doctor("my_video.mp4")
    >>> print(report)
"""

from avdec.doctor.report import DiagnosticReport
from avdec.doctor.analyzer import analyze

__all__ = ["doctor", "DiagnosticReport"]


def doctor(path: str) -> DiagnosticReport:
    """Diagnose video file for ML training suitability.

    Analyzes a video and reports issues that may affect training:
    - How fast/slow random frame access will be
    - Whether frame timing is consistent
    - Container/codec efficiency

    Args:
        path: Path to video file

    Returns:
        DiagnosticReport with findings and recommendations
    """
    return analyze(path)

