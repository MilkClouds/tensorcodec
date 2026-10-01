"""Separate fresh decoder opening from window decoding and count file reads."""

import argparse
import gc
import importlib
import json
import platform
import time
from pathlib import Path

import numpy as np


class CountingFile:
    """Count bytes returned to the decoder, including repeated reads."""

    def __init__(self, source):
        self.source = source
        self.bytes_read = 0
        self.read_calls = 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.bytes_read += len(data)
        self.read_calls += 1
        return data

    def seek(self, offset, whence=0):
        return self.source.seek(offset, whence)

    def tell(self):
        return self.source.tell()


def measure(decoder_class, path, seconds, **options):
    with path.open("rb") as file:
        source = CountingFile(file)
        start = time.perf_counter()
        decoder = decoder_class(source, num_ffmpeg_threads=1, **options)
        opened = time.perf_counter()
        open_bytes, open_calls = source.bytes_read, source.read_calls
        try:
            batch = decoder.get_frames_played_at(seconds=seconds)
            decoded = time.perf_counter()
            result = {
                "open_ms": (opened - start) * 1000,
                "window_ms": (decoded - opened) * 1000,
                "open_bytes": open_bytes,
                "window_bytes": source.bytes_read - open_bytes,
                "open_read_calls": open_calls,
                "window_read_calls": source.read_calls - open_calls,
            }
            # Copies and comparisons are outside both measured intervals.
            arrays = tuple(
                np.asarray(value).copy() for value in (batch.data, batch.pts_seconds, batch.duration_seconds)
            )
        finally:
            if hasattr(decoder, "close"):
                decoder.close()
            del decoder
            gc.collect()
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--backends", nargs="+", choices=("tensorcodec", "torchcodec"), default=["tensorcodec"])
    parser.add_argument("--mappings", type=Path, help="Precomputed ffprobe frame JSON; generation is not timed")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--count", type=int, default=11)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.count < 1 or args.runs < 1 or args.step <= 0:
        parser.error("count, runs and step must be positive")
    seconds = [args.start + i * args.step for i in range(args.count)]
    modes = {"exact": {}, "approximate": {"seek_mode": "approximate"}}
    if args.mappings:
        modes["mapped"] = {"custom_frame_mappings": args.mappings.read_bytes()}
    report = {
        "path": str(args.path.resolve()),
        "file_bytes": args.path.stat().st_size,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "num_ffmpeg_threads": 1,
        "seconds": seconds,
        "io_scope": "bytes returned by local file.read; includes rereads, not physical storage traffic",
        "cache": "uncontrolled OS cache; fresh decoder per trial; rotating mode order",
        "backends": {},
    }
    for backend in args.backends:
        module = importlib.import_module(backend)
        cls = importlib.import_module(backend + ".decoders").VideoDecoder
        results = {mode: [] for mode in modes}
        for run in range(args.runs):
            names = list(modes)
            shift = run % len(names)
            decoded = {}
            for mode in names[shift:] + names[:shift]:
                result, decoded[mode] = measure(cls, args.path, seconds, **modes[mode])
                results[mode].append(result)
            for mode in modes:
                equal = all(np.array_equal(a, b) for a, b in zip(decoded["exact"], decoded[mode]))
                results[mode][-1]["window_matches_exact"] = equal
                if mode == "mapped" and not equal:
                    raise RuntimeError(f"{backend}: mapped pixels/PTS/durations differ from exact")
        report["backends"][backend] = {"version": module.__version__, "trials": results}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
