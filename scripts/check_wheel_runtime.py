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
    # AV1 (dav1d); the reference frames come from the generating FFmpeg's own decode.
    av1 = root / "av1.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=64x64:rate=10:duration=1", "-c:v", "libaom-av1", "-cpu-used", "8", "-g", "4",
         "-pix_fmt", "yuv420p", str(av1)],
        check=True,
    )  # fmt: skip
    reference = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(av1), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True,
        check=True,
    ).stdout
    (root / "av1.rgb").write_bytes(reference)
    samples = np.stack([np.arange(8000) % 1000, -(np.arange(8000) % 1000)], axis=1).astype("<i2")
    with wave.open(str(root / "audio.wav"), "wb") as audio:
        audio.setparams((2, 2, 8000, 8000, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())

    for fmt, channels, storage in [("gray12le", 1, "<u2"), ("gray16be", 1, ">u2"), ("rgba", 4, "u1")]:
        values = np.arange(3 * 11 * 19 * channels).reshape(3, 11, 19, channels)
        values = ((values * 7) % (4096 if fmt == "gray12le" else 65536)).astype(storage)
        raw = root / f"{fmt}.raw"
        raw.write_bytes(values.tobytes())
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                fmt,
                "-video_size",
                "19x11",
                "-framerate",
                "10",
                "-i",
                str(raw),
                "-c:v",
                "rawvideo",
                "-pix_fmt",
                fmt,
                "-threads",
                "1",
                str(root / f"{fmt}.nut"),
            ],
            check=True,
        )

    from PIL import Image

    from tensorcodec.decoders import decode_image

    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (29, 37, 3), dtype=np.uint8))
    for codec in ("JPEG", "PNG", "WEBP", "GIF", "AVIF"):
        path = root / f"image.{codec.lower()}"
        image.save(path, format=codec, max_threads=1)
        np.save(root / f"image.{codec.lower()}.npy", decode_image(path))
    for codec in ("WEBP", "GIF", "AVIF"):
        path = root / f"animated.{codec.lower()}"
        image.save(
            path,
            format=codec,
            save_all=True,
            append_images=[image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)],
            duration=100,
            max_threads=1,
        )
        np.save(root / f"animated.{codec.lower()}.npy", decode_image(path))


def check(root: Path):
    import numpy as np

    from tensorcodec.decoders import AudioDecoder, VideoDecoder

    assert all(importlib.util.find_spec(name) is None for name in ("torch", "torchcodec", "av", "PIL"))
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
    expected = np.frombuffer((root / "av1.rgb").read_bytes(), dtype=np.uint8).reshape(10, 64, 64, 3)
    with VideoDecoder(root / "av1.mkv", dimension_order="NHWC") as decoder:
        assert len(decoder) == 10
        batch = decoder.get_frames_at([9, 0, 5])
        np.testing.assert_allclose(batch.pts_seconds, [0.9, 0.0, 0.5])
        diff = np.abs(batch.data.astype(np.int16) - expected[[9, 0, 5]])
        assert diff.max() <= 2, diff.max()  # RGB conversion may round differently from the FFmpeg CLI
    for fmt, channels, storage in [("gray12le", 1, "<u2"), ("gray16be", 1, ">u2"), ("rgba", 4, "u1")]:
        expected = np.frombuffer((root / f"{fmt}.raw").read_bytes(), dtype=storage).reshape(3, 11, 19, channels)
        with VideoDecoder(root / f"{fmt}.nut", output_format="native") as decoder:
            batch = decoder.get_frames_at([2, 0, 2])
            assert batch.pixel_format == fmt and batch.data.dtype.isnative
            np.testing.assert_array_equal(batch.data, expected[[2, 0, 2]].transpose(0, 3, 1, 2))
    from tensorcodec.decoders import decode_image

    for path in root.glob("image.*.npy"):
        np.testing.assert_array_equal(decode_image(path.with_suffix("")), np.load(path))
    for path in root.glob("animated.*.npy"):
        np.testing.assert_array_equal(decode_image(path.with_suffix("")), np.load(path))
    print(f"Wheel decoding passed: Python {platform.python_version()}, {platform.machine()}, {platform.libc_ver()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "check"))
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    {"generate": generate, "check": check}[args.command](args.fixtures)
