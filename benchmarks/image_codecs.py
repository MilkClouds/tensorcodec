"""Compare RGB decoding on a directory of JPEG/PNG/WebP files; run inside a uv venv."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchcodec
from torchcodec.decoders import decode_image as reference

from tensorcodec.decoders import decode_image


def latency(fn):
    for _ in range(3):
        fn()
    start = time.perf_counter()
    fn()
    count = max(2, min(50, round(0.025 / (time.perf_counter() - start))))
    times = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(count):
            fn()
        times.append((time.perf_counter() - start) * 1000 / count)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    print(json.dumps({"opencv": cv2.__version__, "torchcodec": torchcodec.__version__}))
    for path in sorted(args.corpus.iterdir()):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if not (data.startswith((b"\xff\xd8\xff", b"\x89PNG")) or data[8:12] == b"WEBP"):
            continue
        actual, expected = decode_image(data), reference(data).numpy()
        np.testing.assert_array_equal(actual, expected, err_msg=path.name)
        print(
            json.dumps(
                {
                    "file": path.name,
                    "max_error": 0,
                    "tensorcodec_ms": latency(lambda data=data: decode_image(data)),
                    "torchcodec_ms": latency(lambda data=data: reference(data)),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
