"""Benchmark runner — measures FPS and disk I/O per frame.

Two benchmark *scenarios* are executed for each decoder:

1. **random_access** — ``get_frames_played_at`` with *N* random timestamps.
2. **sequential_range** — ``get_frames_played_in_range`` over the full video.

Usage (as a module)::

    python -m benchmarks.runner [OPTIONS]

Examples::

    # Run all available decoders
    python -m benchmarks.runner

    # Specific video, skip I/O measurement
    python -m benchmarks.runner --video /path/to/video.mp4 --no-io

    # Only avdec, 5 repetitions
    python -m benchmarks.runner --decoders avdec --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass


import numpy as np

from benchmarks.decoders import get_decoder, list_available_decoders
from benchmarks.protocol import BenchmarkResult, VideoDecoderProtocol


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    video_path: str
    num_runs: int = 3
    measure_io: bool = True
    num_random_timestamps: int = 50
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Video creation helper
# ---------------------------------------------------------------------------
def create_test_video(output_path: str, num_frames: int = 300, fps: int = 30) -> str:
    """Create a synthetic test video (640×480, libx264)."""
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"testsrc=duration={num_frames / fps}:size=640x480:rate={fps}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except FileNotFoundError:
        pass
    # Fallback: PyAV
    import av

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
    stream.options = {"g": "10"}
    for i in range(num_frames):
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        data[:, :, 0] = (i * 3) % 256
        data[:, :, 1] = np.arange(640)[None, :] * 255 // 640
        data[:, :, 2] = np.arange(480)[:, None] * 255 // 480
        frame = av.VideoFrame.from_ndarray(data, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return output_path


# ---------------------------------------------------------------------------
# Speed-only benchmark
# ---------------------------------------------------------------------------
def _run_random_access(decoder: VideoDecoderProtocol, video_path: str, cfg: BenchmarkConfig) -> BenchmarkResult:
    rng = random.Random(cfg.random_seed)
    duration = decoder.get_video_duration(video_path)
    timestamps = sorted([rng.uniform(0, duration - 0.001) for _ in range(cfg.num_random_timestamps)])

    times: list[float] = []
    num_frames = 0
    for _ in range(cfg.num_runs):
        t0 = time.perf_counter()
        arr = decoder.get_frames_played_at(video_path, timestamps)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        num_frames = arr.shape[0]

    return BenchmarkResult(
        decoder_name=decoder.name, scenario="random_access",
        num_frames=num_frames, elapsed_time=sum(times) / len(times),
    )


def _run_sequential_range(decoder: VideoDecoderProtocol, video_path: str, cfg: BenchmarkConfig) -> BenchmarkResult:
    duration = decoder.get_video_duration(video_path)

    times: list[float] = []
    num_frames = 0
    for _ in range(cfg.num_runs):
        t0 = time.perf_counter()
        arr = decoder.get_frames_played_in_range(video_path, 0.0, duration)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        num_frames = arr.shape[0]

    return BenchmarkResult(
        decoder_name=decoder.name, scenario="sequential_range",
        num_frames=num_frames, elapsed_time=sum(times) / len(times),
    )


# ---------------------------------------------------------------------------
# FUSE I/O benchmark (optional)
# ---------------------------------------------------------------------------
def _run_with_fuse(decoder: VideoDecoderProtocol, video_path: str, cfg: BenchmarkConfig, scenario_fn) -> BenchmarkResult:
    """Run a scenario through a FUSE mount to capture I/O stats."""
    try:
        import pyfuse3
        import trio
    except ImportError:
        print("  [WARN] pyfuse3/trio not installed — skipping I/O measurement")
        return scenario_fn(decoder, video_path, cfg)

    from benchmarks.counting_fs import CountingFS

    source_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.basename(video_path)

    with tempfile.TemporaryDirectory(prefix="fuse_bench_") as mount_point:
        fs = CountingFS(source_dir)
        fuse_options = set(pyfuse3.default_options)
        fuse_options.add("fsname=countingfs")

        async def _run():
            pyfuse3.init(fs, mount_point, fuse_options)
            async with trio.open_nursery() as nursery:
                nursery.start_soon(pyfuse3.main)
                await trio.sleep(0.5)

                fuse_video = os.path.join(mount_point, video_name)
                result = await trio.to_thread.run_sync(
                    lambda: scenario_fn(decoder, fuse_video, cfg)
                )
                stats = fs.get_stats()
                result.io_bytes = stats["bytes"]
                result.io_calls = stats["calls"]
                nursery.cancel_scope.cancel()

            try:
                pyfuse3.close()
            except Exception:
                pass
            return result

        return trio.run(_run)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------
def print_results(results: list[BenchmarkResult], fmt: str = "table") -> None:
    if fmt == "json":
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    has_io = any(r.io_bytes is not None for r in results)
    header = f"{'Decoder':<15} {'Scenario':<20} {'Frames':<8} {'Time(s)':<10} {'FPS':<10}"
    if has_io:
        header += f" {'I/O(MB)':<10} {'B/Frame':<10}"
    print(header)
    print("-" * 90)
    for r in results:
        line = f"{r.decoder_name:<15} {r.scenario:<20} {r.num_frames:<8} {r.elapsed_time:<10.3f} {r.fps:<10.1f}"
        if has_io and r.io_bytes is not None:
            line += f" {r.io_bytes / (1024 * 1024):<10.2f} {r.bytes_per_frame:<10.1f}"
        print(line)
    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark video decoders (FPS + disk I/O)",
    )
    parser.add_argument("--video", "-v", help="Path to video (creates test video if omitted)")
    parser.add_argument("--decoders", "-d", nargs="+", help="Decoder names to benchmark")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Repetitions per scenario")
    parser.add_argument("--no-io", action="store_true", help="Skip FUSE I/O measurement")
    parser.add_argument("--output", "-o", choices=["table", "json"], default="table")
    parser.add_argument("--timestamps", "-t", type=int, default=50, help="Random timestamps to sample")
    parser.add_argument("--list-decoders", action="store_true", help="Print available decoders")
    args = parser.parse_args()

    if args.list_decoders:
        for name in list_available_decoders():
            print(f"  - {name}")
        return

    decoder_names = args.decoders or list_available_decoders()
    video_path = args.video
    tmp_dir = None

    if video_path is None:
        tmp_dir = tempfile.mkdtemp(prefix="bench_video_")
        video_path = os.path.join(tmp_dir, "test_video.mp4")
        print(f"Creating test video: {video_path}")
        create_test_video(video_path)
        print(f"  Size: {os.path.getsize(video_path) / (1024 * 1024):.2f} MB")

    cfg = BenchmarkConfig(
        video_path=video_path,
        num_runs=args.runs,
        measure_io=not args.no_io,
        num_random_timestamps=args.timestamps,
    )

    results: list[BenchmarkResult] = []
    for name in decoder_names:
        print(f"\nBenchmarking: {name}")
        try:
            dec = get_decoder(name)
            for scenario_fn in (_run_random_access, _run_sequential_range):
                if cfg.measure_io:
                    res = _run_with_fuse(dec, video_path, cfg, scenario_fn)
                else:
                    res = scenario_fn(dec, video_path, cfg)
                results.append(res)
                print(f"  {res.scenario}: {res.fps:.1f} FPS", end="")
                if res.io_bytes is not None:
                    print(f"  I/O {res.io_bytes / (1024 * 1024):.2f} MB", end="")
                print()
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print_results(results, args.output)

    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
