"""Container & codec robustness benchmark.

Generates synthetic videos in various codec × container combinations and
measures random-seek performance for tensorcodec and (optionally) TorchCodec.

Usage::

    uv run --no-sync python -m benchmarks.container_bench
    uv run --no-sync python -m benchmarks.container_bench --duration 60 --queries 200
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

COMBOS: list[tuple[str, str, str, str, dict]] = [
    # (codec_label, extension, av_codec, pix_fmt, encoder_opts)
    ("h264", "mp4", "libx264", "yuv420p", {"g": "10"}),
    ("h264", "mkv", "libx264", "yuv420p", {"g": "10"}),
    ("h264", "avi", "libx264", "yuv420p", {"g": "10"}),
    ("h264", "ts", "libx264", "yuv420p", {"g": "10"}),
    ("h265", "mp4", "libx265", "yuv420p", {"g": "10", "x265-params": "log-level=error"}),
    ("h265", "mkv", "libx265", "yuv420p", {"g": "10", "x265-params": "log-level=error"}),
    ("h265", "ts", "libx265", "yuv420p", {"g": "10", "x265-params": "log-level=error"}),
    ("vp9", "mkv", "libvpx-vp9", "yuv420p", {"g": "10", "quality": "realtime", "speed": "8"}),
    ("vp9", "webm", "libvpx-vp9", "yuv420p", {"g": "10", "quality": "realtime", "speed": "8"}),
    ("mpeg4", "mp4", "mpeg4", "yuv420p", {"g": "10"}),
    ("mpeg4", "mkv", "mpeg4", "yuv420p", {"g": "10"}),
    ("mpeg4", "avi", "mpeg4", "yuv420p", {"g": "10"}),
    ("mjpeg", "mkv", "mjpeg", "yuvj420p", {}),
    ("mjpeg", "avi", "mjpeg", "yuvj420p", {}),
]


def _create_video(
    path: str,
    codec: str,
    pix_fmt: str = "yuv420p",
    duration_sec: int = 30,
    fps: int = 30,
    opts: dict | None = None,
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=640x480:rate={fps}:duration={duration_sec}",
        "-c:v",
        codec,
        "-pix_fmt",
        pix_fmt,
    ]
    for key, value in (opts or {}).items():
        cmd.extend([f"-{key}", str(value)])
    subprocess.run([*cmd, path], capture_output=True, check=True)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench_tensorcodec(path: str, windows: list[list[float]]) -> str:
    from tensorcodec.decoders import VideoDecoder

    try:
        with VideoDecoder(path) as d:
            d.get_frames_played_at([1.0])
    except (RuntimeError, ValueError, OSError):
        return "FAIL"
    t0 = time.perf_counter()
    total = 0
    for w in windows:
        with VideoDecoder(path) as d:
            total += d.get_frames_played_at(w).data.shape[0]
    return f"{total / (time.perf_counter() - t0):.0f}"


def _bench_torchcodec(path: str, windows: list[list[float]], seek_mode: str) -> str:
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError:
        return "N/A"
    try:
        VideoDecoder(path, seek_mode=seek_mode, num_ffmpeg_threads=1).get_frames_played_at([1.0])
    except (RuntimeError, ValueError, OSError):
        return "FAIL"
    t0 = time.perf_counter()
    total = 0
    for w in windows:
        dec = VideoDecoder(path, seek_mode=seek_mode, num_ffmpeg_threads=1)
        total += dec.get_frames_played_at(w).data.shape[0]
    return f"{total / (time.perf_counter() - t0):.0f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Container robustness benchmark")
    parser.add_argument("--duration", type=int, default=30, help="Video duration in seconds")
    parser.add_argument("--queries", type=int, default=100, help="Number of random-seek queries")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    lo = max(2.0, args.duration * 0.1)
    hi = args.duration - 1.0
    ts = [random.uniform(lo, hi) for _ in range(args.queries)]
    windows = [[t + o for o in np.arange(-1.0, 0.05, 0.1).tolist()] for t in ts]

    with tempfile.TemporaryDirectory(prefix="container_bench_") as tmp:
        hdr = (
            f"{'codec.container':20s} {'tensorcodec':>10s} {'TC(apx)':>10s} {'TC(ext)':>10s}  {'apx/tensorcodec':>10s}"
        )
        print(hdr)
        print("-" * len(hdr))

        for codec_label, ext, av_codec, pix, opts in COMBOS:
            path = os.path.join(tmp, f"{codec_label}.{ext}")
            try:
                _create_video(path, av_codec, pix, args.duration, opts=opts)
            except (OSError, subprocess.CalledProcessError) as e:
                print(f"{codec_label}.{ext:20s} CREATE FAILED: {e}")
                continue

            r1 = _bench_tensorcodec(path, windows)
            r2 = _bench_torchcodec(path, windows, "approximate")
            r3 = _bench_torchcodec(path, windows, "exact")
            try:
                ratio = f"{float(r2) / float(r1):.2f}x"
            except (ValueError, ZeroDivisionError):
                ratio = "N/A"
            label = f"{codec_label}.{ext}"
            print(f"{label:20s} {r1:>10s} {r2:>10s} {r3:>10s}  {ratio:>10s}")


if __name__ == "__main__":
    main()
