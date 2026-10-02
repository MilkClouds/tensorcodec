"""Compare complete bytes-to-RGB NumPy decoding; optional Rust backends are isolated."""

import argparse
import importlib.metadata
import io
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchcodec import decoders as oracle

from tensorcodec import decoders


def inputs(root):
    source = root / "tests/resources/nasa_13013.mp4"
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-ss",
            "1",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-threads",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    photo = Image.open(io.BytesIO(encoded)).convert("RGB")
    for w, h in [(224, 224), (640, 480), (1920, 1080)]:
        y, x = np.indices((h, w))
        patterns = {
            "photo": photo.resize((w, h), Image.Resampling.LANCZOS),
            "graphics": Image.fromarray(
                np.stack(((x // 17 % 2) * 255, (y // 23 % 2) * 255, ((x + y) // 31 % 2) * 255), axis=-1).astype(
                    "uint8"
                )
            ),
            "noise": Image.fromarray(np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8)),
        }
        formats = [
            ("JPEG420", "JPEG", {"quality": 90, "subsampling": 2}),
            ("JPEG444", "JPEG", {"quality": 90, "subsampling": 0}),
            ("JPEG-progressive", "JPEG", {"quality": 90, "progressive": True}),
            ("PNG", "PNG", {}),
            ("WebP", "WEBP", {"quality": 90}),
            ("WebP-lossless", "WEBP", {"lossless": True}),
        ]
        for content, image in patterns.items():
            for name, fmt, options in formats:
                buffer = io.BytesIO()
                image.save(buffer, format=fmt, **options)
                yield f"{content}-{w}x{h}-{name}", name, content, (w, h), buffer.getvalue()


def measure(calls):
    repeats, times = {}, {name: [] for name in calls}
    for name, fn in calls.items():
        for _ in range(3):
            fn()
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        repeats[name] = max(2, min(50, round(0.035 / max(elapsed, 1e-6))))
    for batch in range(5):
        keys = list(calls)
        keys = keys[batch % len(keys) :] + keys[: batch % len(keys)]
        for name in keys:
            start = time.perf_counter()
            for _ in range(repeats[name]):
                calls[name]()
            times[name].append((time.perf_counter() - start) * 1000 / repeats[name])
    return {name: statistics.median(values) for name, values in times.items()}


def run(output, candidates):
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rust = None
    if candidates:
        import _image_candidate as rust
    report = {
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "torch", "torchcodec")},
        "method": "RGB uint8 CHW NumPy; in-memory bytes; one CPU affinity where available; 3 warmups; "
        "5 rotated-order batches, median milliseconds; allocation/materialization included, encoding/IO excluded.",
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for case, fmt, content, size, data in inputs(Path(__file__).resolve().parents[1]):

        def pillow(data=data):
            with Image.open(io.BytesIO(data)) as image:
                return np.asarray(image.convert("RGB")).transpose(2, 0, 1)

        calls = {
            "tensorcodec": lambda data=data: decoders.decode_image(data),
            "torchcodec": lambda data=data: oracle.decode_image(data).numpy(),
            "pillow": pillow,
        }
        if rust:
            calls["image-rs"] = lambda data=data: rust.decode(data).transpose(2, 0, 1)
            if fmt.startswith("JPEG"):
                calls["turbo-rs"] = lambda data=data: rust.decode_turbo(data).transpose(2, 0, 1)
        values = {name: fn() for name, fn in calls.items()}
        diffs = {}
        for name, value in values.items():
            delta = np.abs(value.astype(np.int16) - values["torchcodec"].astype(np.int16))
            diffs[name] = {"max": int(delta.max()), "mae": float(delta.mean())}
        row = {
            "case": case,
            "format": fmt,
            "content": content,
            "size": size,
            "bytes": len(data),
            "ms": measure(calls),
            "diff_vs_torch": diffs,
        }
        report["rows"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", action="store_true")
    args = parser.parse_args()
    run(args.output, args.candidates)
