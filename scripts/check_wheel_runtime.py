"""Generate portable fixtures, then decode them in a clean wheel runtime."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import subprocess
import wave
from pathlib import Path


def generate(root: Path):
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-g",
            "10",
            "-bf",
            "2",
            str(root / "video.mp4"),
        ],
        check=True,
    )
    samples = np.stack([np.arange(8000) % 1000, -(np.arange(8000) % 1000)], axis=1).astype("<i2")
    with wave.open(str(root / "audio.wav"), "wb") as audio:
        audio.setparams((2, 2, 8000, 8000, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())


def check(root: Path):
    import numpy as np

    from tensorcodec.decoders import AudioDecoder, VideoDecoder

    assert all(importlib.util.find_spec(name) is None for name in ("torch", "torchcodec", "av"))
    with VideoDecoder(root / "video.mp4") as decoder:
        assert len(decoder) == 10
        frames = decoder.get_frames_at([7, 0, 7])
        assert frames.data.shape == (3, 3, 48, 64)
        assert frames.data.dtype == np.uint8
        np.testing.assert_allclose(frames.pts_seconds, [0.7, 0.0, 0.7])
        np.testing.assert_array_equal(frames.data[0], frames.data[2])
        played = decoder.get_frame_played_at(0.75)
        np.testing.assert_array_equal(played.data, frames.data[0])
    assert frames.data.max() > frames.data.min()
    with AudioDecoder(root / "audio.wav") as decoder:
        audio = decoder.get_samples_played_in_range(0.125, 0.25)
        assert audio.data.shape == (2, 1000)
        assert audio.data.dtype == np.float32
        expected = np.arange(1000, dtype=np.float32) / 32768
        np.testing.assert_allclose(audio.data[0], expected, atol=1e-7)
        np.testing.assert_allclose(audio.data[1], -expected, atol=1e-7)
    print(f"Wheel decoding passed: Python {platform.python_version()}, {platform.machine()}, {platform.libc_ver()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "check"))
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    {"generate": generate, "check": check}[args.command](args.fixtures)
