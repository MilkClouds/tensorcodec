"""Array adapters and independent FFmpeg helpers used by decoder tests."""

import json
import subprocess

import numpy as np


def run_ffmpeg(*args):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True,
        capture_output=True,
    )


def as_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    return value.numpy()


def index_input(backend, values):
    if backend.__name__.startswith("torchcodec"):
        import torch

        return torch.tensor(values, dtype=torch.int64)
    return np.asarray(values, dtype=np.int64)


def time_input(backend, values):
    if backend.__name__.startswith("torchcodec"):
        import torch

        return torch.tensor(values, dtype=torch.float64)
    return np.asarray(values, dtype=np.float64)


def probe_stream(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def ffmpeg_rgb(path, pixel_format, shape):
    output = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-threads",
            "1",
            "-sws_flags",
            "0",
            "-pix_fmt",
            pixel_format,
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(output, dtype=np.uint8 if pixel_format == "rgb24" else "<u2").reshape(shape)
