"""Video analysis and issue detection."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import List

import av

from avdec.doctor.report import DiagnosticReport, Issue, Severity, VideoStats


def analyze(path: str) -> DiagnosticReport:
    """Analyze a video file and generate diagnostic report."""
    filepath = Path(path)

    if not filepath.exists():
        return DiagnosticReport(error=f"File not found: {path}")

    try:
        stats = _extract_stats(filepath)
        issues = _detect_issues(stats, filepath)
        return DiagnosticReport(stats=stats, issues=issues)
    except Exception as e:
        return DiagnosticReport(error=str(e))


def _extract_stats(filepath: Path) -> VideoStats:
    """Extract video statistics using PyAV."""
    container = av.open(str(filepath))
    stream = container.streams.video[0]

    # Basic info
    duration = float(container.duration / av.time_base) if container.duration else 0
    fps = float(stream.average_rate) if stream.average_rate else 0
    frame_count = stream.frames or int(duration * fps)

    stats = VideoStats(
        path=filepath,
        container=container.format.name,
        codec=stream.codec_context.name,
        width=stream.width,
        height=stream.height,
        duration_sec=duration,
        frame_count=frame_count,
        fps_avg=fps,
        file_size_bytes=filepath.stat().st_size,
    )

    # Analyze packets for keyframes and timing
    _analyze_packets(container, stream, stats)

    # Check moov position for MP4
    if stats.container in ("mov", "mp4", "m4a", "3gp"):
        stats.moov_position = _check_moov_position(filepath)

    container.close()
    return stats


def _analyze_packets(container: av.InputContainer, stream: av.VideoStream, stats: VideoStats) -> None:
    """Analyze packet-level info: keyframes, frame durations."""
    keyframe_positions: List[int] = []
    pts_values: List[float] = []

    time_base = float(stream.time_base)
    frame_idx = 0

    for packet in container.demux(stream):
        if packet.pts is None:
            continue

        pts_sec = packet.pts * time_base
        pts_values.append(pts_sec)

        if packet.is_keyframe:
            keyframe_positions.append(frame_idx)

        frame_idx += 1

    # Keyframe statistics
    stats.keyframe_count = len(keyframe_positions)
    if len(keyframe_positions) >= 2:
        intervals = [keyframe_positions[i + 1] - keyframe_positions[i] for i in range(len(keyframe_positions) - 1)]
        stats.keyframe_interval_frames_avg = statistics.mean(intervals)
        stats.keyframe_interval_frames_max = max(intervals)
        if stats.fps_avg > 0:
            stats.keyframe_interval_sec_avg = stats.keyframe_interval_frames_avg / stats.fps_avg

    # Frame timing statistics - sort PTS to get presentation order durations
    if len(pts_values) > 1:
        sorted_pts = sorted(pts_values)
        frame_durations_ms = [
            (sorted_pts[i + 1] - sorted_pts[i]) * 1000
            for i in range(len(sorted_pts) - 1)
            if (sorted_pts[i + 1] - sorted_pts[i]) > 0
        ]

        if frame_durations_ms:
            stats.frame_duration_min_ms = min(frame_durations_ms)
            stats.frame_duration_max_ms = max(frame_durations_ms)
            if len(frame_durations_ms) > 1:
                stats.frame_duration_stddev_ms = statistics.stdev(frame_durations_ms)

            # Check for VFR (variable frame rate)
            expected_duration = 1000 / stats.fps_avg if stats.fps_avg > 0 else 0
            if expected_duration > 0:
                variation = stats.frame_duration_stddev_ms / expected_duration
                stats.is_cfr = variation < 0.1  # <10% variation = CFR

    container.seek(0)


def _check_moov_position(filepath: Path) -> str:
    """Check if moov atom is at the start or end of MP4 file."""
    try:
        with open(filepath, "rb") as f:
            # Read first 64KB to find moov
            header = f.read(65536)
            if b"moov" in header[:32768]:
                return "start"

            # Check end of file
            f.seek(-65536, 2)
            tail = f.read()
            if b"moov" in tail:
                return "end"
    except Exception:
        pass
    return "unknown"


def _detect_issues(stats: VideoStats, filepath: Path) -> List[Issue]:
    """Detect issues based on video statistics."""
    issues: List[Issue] = []

    issues.extend(_check_keyframe_interval(stats, filepath))
    issues.extend(_check_frame_rate_consistency(stats, filepath))
    issues.extend(_check_moov_location(stats, filepath))

    return issues


def _check_keyframe_interval(stats: VideoStats, filepath: Path) -> List[Issue]:
    """Check if keyframe interval makes random access slow."""
    issues = []

    # Handle case where there's only 1 keyframe (entire video is one GOP)
    if stats.keyframe_count <= 1 and stats.frame_count > 60:
        avg_decode_count = stats.frame_count / 2
        decode_time_per_frame_ms = 1.0
        random_access_ms = avg_decode_count * decode_time_per_frame_ms

        body = f"""
   This video has only 1 keyframe.
   To read any frame, you must decode from the beginning.

   Estimated time:
   ├─ First frame access: ~1ms
   └─ Last frame access: ~{int(stats.frame_count)}ms (decoding {stats.frame_count} frames)

   What is a keyframe?
   └─ A point in the video you can jump to directly.
      Other frames must be decoded sequentially from the nearest keyframe.
      Currently, only the first frame is a keyframe, so all random access starts from the beginning.

   Recommendation: Re-encode with keyframe interval of 1 second (~30 frames)""".strip()

        fix_cmd = f'ffmpeg -i "{filepath.name}" -c:v libx264 -g 30 -c:a copy "{filepath.stem}_fixed.mp4"'

        issues.append(
            Issue(
                severity=Severity.ERROR,
                title="Almost no keyframes (random access very slow)",
                body=body,
                fix_command=fix_cmd,
            )
        )
        return issues

    avg_interval = stats.keyframe_interval_frames_avg
    max_interval = stats.keyframe_interval_frames_max

    if avg_interval <= 0:
        return issues

    # Estimate random access cost
    avg_decode_count = avg_interval / 2
    decode_time_per_frame_ms = 1.0
    random_access_ms = avg_decode_count * decode_time_per_frame_ms
    interval_sec = stats.keyframe_interval_sec_avg

    if avg_interval > 60:  # More than 2 seconds at 30fps
        severity = Severity.WARN if avg_interval <= 150 else Severity.ERROR

        body = f"""
   To read a random frame from this video,
   you must first decode {int(avg_decode_count)} frames on average.

   Estimated time:
   ├─ Sequential access (0->1->2): ~1ms / frame
   └─ Random access (0->500->123): ~{int(random_access_ms)}ms / frame

   Cause: Keyframe interval = {int(avg_interval)} frames ({interval_sec:.1f} sec)
         │
         ├─ What is a keyframe?
         │  A point in the video you can jump to directly.
         │  Other frames must be decoded sequentially from the nearest keyframe.
         │
         └─ Current state
            Keyframes occur only every {int(interval_sec)} seconds,
            so accessing frames in between requires decoding up to {max_interval} frames.

   Recommendation: For training, re-encode with keyframe interval of 1 sec (~30 frames)""".strip()

        fix_cmd = f'ffmpeg -i "{filepath.name}" -c:v libx264 -g 30 -c:a copy "{filepath.stem}_fixed.mp4"'

        issues.append(
            Issue(
                severity=severity,
                title="Random frame access is slow",
                body=body,
                fix_command=fix_cmd,
            )
        )

    return issues


def _check_frame_rate_consistency(stats: VideoStats, filepath: Path) -> List[Issue]:
    """Check for variable frame rate issues."""
    issues = []

    if stats.is_cfr:
        return issues

    min_ms = stats.frame_duration_min_ms
    max_ms = stats.frame_duration_max_ms
    stddev = stats.frame_duration_stddev_ms
    expected_ms = 1000 / stats.fps_avg if stats.fps_avg > 0 else 0

    body = f"""
   Frame intervals are inconsistent (VFR: Variable Frame Rate).

   Frame interval distribution:
   ├─ Min: {min_ms:.1f}ms
   ├─ Max: {max_ms:.1f}ms
   ├─ Std dev: {stddev:.1f}ms
   └─ Expected ({stats.fps_avg:.0f}fps): {expected_ms:.1f}ms

   Why is this a problem?
   ├─ "Frame 100" doesn't always mean the same point in time.
   ├─ Frame index-based sampling becomes uneven.
   └─ Hard to compare frame numbers across different videos.

   Cause: Smartphone recording, screen capture, or variable bitrate encoding

   Recommendation: Use timestamp-based access (get_frames_played_at),
                   or re-encode to constant frame rate""".strip()

    fix_cmd = f'ffmpeg -i "{filepath.name}" -vf "fps=30" -c:a copy "{filepath.stem}_cfr.mp4"'

    issues.append(
        Issue(
            severity=Severity.WARN,
            title="Irregular frame intervals (VFR)",
            body=body,
            fix_command=fix_cmd,
        )
    )

    return issues


def _check_moov_location(stats: VideoStats, filepath: Path) -> List[Issue]:
    """Check if MP4 moov atom is at the end (bad for streaming/seeking)."""
    issues = []

    if stats.container not in ("mov", "mp4"):
        return issues

    if stats.moov_position == "end":
        body = f"""
   This MP4 file's "table of contents" (moov atom) is at the end of the file.

   Impact:
   ├─ Opening the file requires scanning the entire file.
   ├─ Especially slow on network storage.
   └─ Larger files ({stats.file_size_mb:.0f}MB) have more delay.

   Analogy:
   └─ Like a book with the table of contents at the very end,
      you must flip to the last page every time to find what you need.

   Cause: faststart option not applied during encoding""".strip()

        fix_cmd = f'ffmpeg -i "{filepath.name}" -c copy -movflags +faststart "{filepath.stem}_faststart.mp4"'

        issues.append(
            Issue(
                severity=Severity.WARN,
                title="MP4 metadata location is inefficient",
                body=body,
                fix_command=fix_cmd,
            )
        )

    return issues
