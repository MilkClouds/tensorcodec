"""Benchmark runner — measures FPS and (optionally) disk I/O per frame.

Two benchmark *scenarios* are executed for each decoder across all videos:

1. **temporal_window** — VLA-style access: pick random "current"
   timestamps across the video corpus, then for each read a window of
   frames at ``[t-1.0, t-0.9, …, t-0.1, t]`` (11 frames per query).
   This mirrors real VLA training where an agent reads the last ~1 s
   of video at every step.
2. **sequential_range** — ``get_frames_played_in_range`` over each full
   video in the corpus.

Video preparation is separated from benchmarking so that the same corpus
can be reused across runs.

Usage::

    # 1. Prepare videos (only needed once)
    python -m benchmarks --prepare

    # 2. Run benchmark (reuses prepared videos)
    python -m benchmarks --no-io

    # Custom corpus
    python -m benchmarks --prepare --num-videos 16 --video-duration 120
    python -m benchmarks --no-io --decoders avdec torchcodec
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from benchmarks.decoders import get_decoder, list_available_decoders
from benchmarks.protocol import BenchmarkResult, VideoDecoderProtocol

DEFAULT_VIDEO_DIR = Path(__file__).resolve().parent.parent / ".bench_videos"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class VideoCorpusConfig:
    """Parameters for generating the test video corpus."""

    video_dir: Path = field(default_factory=lambda: DEFAULT_VIDEO_DIR)
    num_videos: int = 64
    duration_sec: float = 300.0
    fps: int = 30
    keyframe_interval: int = 10


@dataclass
class BenchmarkConfig:
    """Parameters that control how benchmarks are executed."""

    num_runs: int = 3
    measure_io: bool = True
    num_queries_per_video: int = 5
    window_seconds: float = 1.0
    window_step: float = 0.1
    random_seed: int = 42
    timeout_seconds: float = 30.0
    warmup_queries: int = 3


# ---------------------------------------------------------------------------
# Video preparation (separated from benchmarking)
# ---------------------------------------------------------------------------
def _create_one_video(
    output_path: str,
    duration_sec: float,
    fps: int = 30,
    keyframe_interval: int = 10,
) -> None:
    """Create a synthetic video (640×480, libx264)."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_sec}:size=640x480:rate={fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(keyframe_interval),
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return
    except FileNotFoundError:
        pass
    # Fallback: PyAV
    import av

    num_frames = int(duration_sec * fps)
    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
    stream.options = {"g": str(keyframe_interval)}
    for i in range(num_frames):
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        data[:, :, 0] = (i * 3) % 256
        data[:, :, 1] = np.arange(640)[None, :] * 255 // 640
        data[:, :, 2] = np.arange(480)[:, None] * 255 // 480
        frame_obj = av.VideoFrame.from_ndarray(data, format="rgb24")
        for pkt in stream.encode(frame_obj):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def prepare_videos(vcfg: VideoCorpusConfig) -> list[str]:
    """Create synthetic videos according to *vcfg*.

    Existing videos are kept — only missing ones are generated.
    Returns the list of video paths.
    """
    vcfg.video_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    total_duration_min = vcfg.num_videos * vcfg.duration_sec / 60

    print(f"Video corpus: {vcfg.num_videos} × {vcfg.duration_sec:.0f}s = {total_duration_min:.0f} min")
    print(f"  Directory : {vcfg.video_dir}")
    print(f"  Params    : {vcfg.fps}fps, 640×480, libx264, keyframe_interval={vcfg.keyframe_interval}")

    for idx in range(vcfg.num_videos):
        name = f"video_{idx:04d}.mp4"
        path = vcfg.video_dir / name
        paths.append(str(path))
        if path.exists():
            continue
        print(f"  Creating {name} ({idx + 1}/{vcfg.num_videos}) ...", end="", flush=True)
        _create_one_video(str(path), vcfg.duration_sec, vcfg.fps, vcfg.keyframe_interval)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f" {size_mb:.1f} MB")

    total_mb = sum(Path(p).stat().st_size for p in paths) / (1024 * 1024)
    print(f"  Total size: {total_mb:.1f} MB  ({len(paths)} videos ready)")
    return paths


def discover_videos(video_dir: Path) -> list[str]:
    """Return sorted list of .mp4 files in *video_dir*."""
    if not video_dir.exists():
        return []
    return sorted(str(p) for p in video_dir.glob("video_*.mp4"))


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------
_CallbackFn = Callable[[], None]


def _run_temporal_window(
    decoder: VideoDecoderProtocol,
    video_paths: list[str],
    cfg: BenchmarkConfig,
    on_warmup_done: _CallbackFn | None = None,
) -> BenchmarkResult:
    """VLA-style access across the video corpus."""
    rng = random.Random(cfg.random_seed)
    offsets = list(np.arange(-cfg.window_seconds, cfg.window_step * 0.5, cfg.window_step))

    # Pre-build queries: (video_path, [timestamps])
    queries: list[tuple[str, list[float]]] = []
    for vp in video_paths:
        duration = decoder.get_video_duration(vp)
        for _ in range(cfg.num_queries_per_video):
            t_now = rng.uniform(cfg.window_seconds, duration - 0.001)
            queries.append((vp, [t_now + off for off in offsets]))

    # Warmup: run a few queries to prime page cache / JIT / libraries
    for vp, ts_window in queries[: cfg.warmup_queries]:
        decoder.get_frames_played_at(vp, ts_window)

    # Signal that warmup is done — allows FUSE stats to be reset so
    # I/O measurement only covers the timed runs.
    if on_warmup_done is not None:
        on_warmup_done()

    deadline = cfg.timeout_seconds
    times: list[float] = []
    total_frames = 0
    timed_out = False
    for run_idx in range(cfg.num_runs):
        t0 = time.perf_counter()
        run_frames = 0
        for vp, ts_window in queries:
            arr = decoder.get_frames_played_at(vp, ts_window)
            run_frames += arr.shape[0]
            if time.perf_counter() - t0 >= deadline:
                timed_out = True
                break
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        total_frames += run_frames
        if timed_out:
            break

    avg_frames = round(total_frames / len(times))
    return BenchmarkResult(
        decoder_name=decoder.name,
        scenario="temporal_window",
        num_frames=avg_frames,
        elapsed_time=sum(times) / len(times),
        timed_out=timed_out,
        num_runs_completed=len(times),
    )


def _run_sequential_range(
    decoder: VideoDecoderProtocol,
    video_paths: list[str],
    cfg: BenchmarkConfig,
    on_warmup_done: _CallbackFn | None = None,
) -> BenchmarkResult:
    """Decode every video start-to-end."""
    # Pre-compute durations so we don't open each file twice inside the loop
    durations = {vp: decoder.get_video_duration(vp) for vp in video_paths}

    # Warmup: decode a short range from the first video
    if video_paths:
        decoder.get_frames_played_in_range(video_paths[0], 0.0, min(1.0, durations[video_paths[0]]))

    # Signal that warmup is done — allows FUSE stats to be reset so
    # I/O measurement only covers the timed runs.
    if on_warmup_done is not None:
        on_warmup_done()

    deadline = cfg.timeout_seconds
    times: list[float] = []
    total_frames = 0
    timed_out = False
    for run_idx in range(cfg.num_runs):
        t0 = time.perf_counter()
        run_frames = 0
        for vp in video_paths:
            arr = decoder.get_frames_played_in_range(vp, 0.0, durations[vp])
            run_frames += arr.shape[0]
            if time.perf_counter() - t0 >= deadline:
                timed_out = True
                break
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        total_frames += run_frames
        if timed_out:
            break

    avg_frames = round(total_frames / len(times))
    return BenchmarkResult(
        decoder_name=decoder.name,
        scenario="sequential_range",
        num_frames=avg_frames,
        elapsed_time=sum(times) / len(times),
        timed_out=timed_out,
        num_runs_completed=len(times),
    )


# ---------------------------------------------------------------------------
# FUSE I/O measurement
# ---------------------------------------------------------------------------
# NOTE: Running through FUSE adds measurable overhead to FPS (14–68%
# depending on the decoder).  Use ``--no-io`` for accurate speed numbers.
# The default mode (with FUSE) is intended for I/O measurement only;
# the ``--verify-fuse`` flag can quantify the overhead for a given setup.


def _run_with_fuse(
    decoder: VideoDecoderProtocol,
    video_paths: list[str],
    cfg: BenchmarkConfig,
    scenario_fn: Callable,
) -> BenchmarkResult:
    """Run *scenario_fn* through a FUSE mount to capture I/O stats.

    Mounts the video directory via :class:`CountingFS`, rewrites all
    video paths to go through the mount, runs the scenario, then
    collects cumulative read-byte / read-call stats.

    The scenario function receives an ``on_warmup_done`` callback that
    resets FUSE counters after warmup, so only the timed measurement
    runs contribute to I/O stats.  The raw FUSE totals are then
    divided by ``result.num_runs_completed`` to produce per-run I/O.
    """
    try:
        import pyfuse3
        import trio
    except ImportError:
        print("  [WARN] pyfuse3/trio not installed — skipping I/O measurement")
        return scenario_fn(decoder, video_paths, cfg)

    from benchmarks.counting_fs import CountingFS

    # All videos share a common parent directory
    source_dir = str(Path(video_paths[0]).resolve().parent)

    with tempfile.TemporaryDirectory(prefix="fuse_bench_") as mount_point:
        fs = CountingFS(source_dir)
        fuse_options = set(pyfuse3.default_options)
        fuse_options.add("fsname=countingfs")

        # Rewrite paths to go through the FUSE mount
        fuse_paths = [os.path.join(mount_point, os.path.basename(vp)) for vp in video_paths]

        async def _run():
            pyfuse3.init(fs, mount_point, fuse_options)
            try:
                async with trio.open_nursery() as nursery:
                    nursery.start_soon(pyfuse3.main)
                    await trio.sleep(0.5)  # let FUSE mount settle

                    # Pass a callback that resets FUSE stats after warmup
                    # so I/O measurement excludes warmup / metadata reads.
                    result = await trio.to_thread.run_sync(
                        lambda: scenario_fn(
                            decoder,
                            fuse_paths,
                            cfg,
                            on_warmup_done=fs.reset_stats,
                        )
                    )
                    stats = fs.get_stats()

                    # Normalize I/O by the number of completed runs so
                    # bytes_per_frame = per-run I/O / per-run frames.
                    n_runs = max(result.num_runs_completed, 1)
                    result.io_bytes = stats["bytes"] // n_runs
                    result.io_calls = stats["calls"] // n_runs

                    nursery.cancel_scope.cancel()
            finally:
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
    header = f"{'Decoder':<20} {'Scenario':<20} {'Frames':<8} {'Time(s)':<10} {'FPS':<10}"
    if has_io:
        header += f" {'I/O(MB)':<10} {'B/Frame':<10}"
    print(header)
    print("-" * 90)
    for r in results:
        timeout_mark = " *" if r.timed_out else ""
        line = f"{r.decoder_name:<20} {r.scenario:<20} {r.num_frames:<8} {r.elapsed_time:<10.3f} {r.fps:<10.1f}"
        if has_io and r.io_bytes is not None:
            line += f" {r.io_bytes / (1024 * 1024):<10.2f} {r.bytes_per_frame:<10.1f}"
        line += timeout_mark
        print(line)
    print("=" * 90)
    if any(r.timed_out for r in results):
        print("* = timed out (partial result)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark video decoders (FPS + disk I/O)",
    )
    # Preparation
    parser.add_argument("--prepare", action="store_true", help="Generate test videos and exit (reuse with later runs)")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help=f"Directory for video corpus (default: {DEFAULT_VIDEO_DIR})",
    )
    parser.add_argument("--num-videos", type=int, default=64, help="Number of test videos to generate (default: 64)")
    parser.add_argument(
        "--video-duration", type=float, default=300.0, help="Duration of each video in seconds (default: 300 = 5 min)"
    )
    parser.add_argument(
        "--keyframe-interval", type=int, default=10, help="Keyframe (GOP) interval in frames (default: 10)"
    )
    # Benchmarking
    parser.add_argument("--decoders", "-d", nargs="+", help="Decoder names to benchmark")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Repetitions per scenario")
    parser.add_argument("--no-io", action="store_true", help="Skip FUSE I/O measurement")
    parser.add_argument(
        "--verify-fuse", action="store_true", help="Run each scenario with AND without FUSE to measure FUSE overhead"
    )
    parser.add_argument("--output", "-o", choices=["table", "json"], default="table")
    parser.add_argument(
        "--save", type=Path, default=None, help="Save results to a JSON file (e.g. results/speed.json)"
    )
    parser.add_argument("--queries", "-q", type=int, default=5, help="Temporal-window queries per video (default: 5)")
    parser.add_argument("--window", type=float, default=1.0, help="Temporal window length in seconds (default: 1.0)")
    parser.add_argument("--window-step", type=float, default=0.1, help="Step between frames in window (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=30.0,
        help="Timeout per scenario run in seconds (default: 30). Partial results are kept on timeout.",
    )
    parser.add_argument(
        "--scenarios",
        "-s",
        nargs="+",
        choices=["temporal_window", "sequential_range"],
        help="Scenarios to run (default: all)",
    )
    parser.add_argument("--list-decoders", action="store_true", help="Print available decoders")
    args = parser.parse_args()

    if args.list_decoders:
        for name in list_available_decoders():
            print(f"  - {name}")
        return

    # --- Build configs from CLI args ---
    vcfg = VideoCorpusConfig(
        video_dir=args.video_dir,
        num_videos=args.num_videos,
        duration_sec=args.video_duration,
        keyframe_interval=args.keyframe_interval,
    )

    # --- Prepare ---
    if args.prepare:
        prepare_videos(vcfg)
        return

    # --- Discover videos ---
    video_paths = discover_videos(vcfg.video_dir)
    if not video_paths:
        print(f"No videos found in {vcfg.video_dir}")
        print("Run with --prepare first to generate the test corpus.")
        sys.exit(1)

    print(f"Video corpus: {len(video_paths)} videos in {vcfg.video_dir}")

    cfg = BenchmarkConfig(
        num_runs=args.runs,
        measure_io=not args.no_io,
        num_queries_per_video=args.queries,
        window_seconds=args.window,
        window_step=args.window_step,
        random_seed=args.seed,
        timeout_seconds=args.timeout,
    )

    decoder_names = args.decoders or list_available_decoders()
    results: list[BenchmarkResult] = []

    # Build scenario list
    _ALL_SCENARIOS = {
        "temporal_window": _run_temporal_window,
        "sequential_range": _run_sequential_range,
    }
    if args.scenarios:
        scenario_fns = [(n, _ALL_SCENARIOS[n]) for n in args.scenarios]
    else:
        scenario_fns = list(_ALL_SCENARIOS.items())

    def _fmt_result_line(res: BenchmarkResult) -> str:
        """Format a single result for inline progress output."""
        parts = [f"  {res.scenario}: {res.fps:.1f} FPS"]
        if res.io_bytes is not None:
            parts.append(f"  I/O {res.io_bytes / (1024 * 1024):.1f} MB  ({res.bytes_per_frame:.0f} B/frame)")
        if res.timed_out:
            parts.append(f"  [TIMEOUT after {res.elapsed_time:.1f}s, {res.num_frames} frames]")
        return "".join(parts)

    if args.verify_fuse:
        # --- FUSE overhead verification ---
        # Run each scenario twice: direct (no FUSE) and through FUSE.
        print("\n*** FUSE overhead verification mode ***")
        for name in decoder_names:
            print(f"\nBenchmarking: {name}")
            try:
                dec = get_decoder(name)
                for _sname, scenario_fn in scenario_fns:
                    # Direct (no FUSE)
                    res_direct = scenario_fn(dec, video_paths, cfg)
                    res_direct.scenario = res_direct.scenario + " [direct]"
                    results.append(res_direct)
                    print(_fmt_result_line(res_direct))

                    # Through FUSE
                    res_fuse = _run_with_fuse(dec, video_paths, cfg, scenario_fn)
                    res_fuse.scenario = res_fuse.scenario + " [fuse]"
                    results.append(res_fuse)
                    overhead = (res_direct.fps - res_fuse.fps) / res_direct.fps * 100
                    print(_fmt_result_line(res_fuse) + f"  (overhead: {overhead:+.1f}%)")
            except Exception as exc:
                print(f"  ERROR: {exc}")
    else:
        # --- Normal benchmark ---
        for name in decoder_names:
            print(f"\nBenchmarking: {name}")
            try:
                dec = get_decoder(name)
                for _sname, scenario_fn in scenario_fns:
                    if cfg.measure_io:
                        res = _run_with_fuse(dec, video_paths, cfg, scenario_fn)
                    else:
                        res = scenario_fn(dec, video_paths, cfg)
                    results.append(res)
                    print(_fmt_result_line(res))
            except Exception as exc:
                print(f"  ERROR: {exc}")

    print_results(results, args.output)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
