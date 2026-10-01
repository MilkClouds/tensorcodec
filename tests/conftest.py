"""Independent media fixtures; PyAV and the old avdec implementation are never used."""

from __future__ import annotations

import importlib
import json
import subprocess
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--backend", choices=("tensorcodec", "torchcodec"), default="tensorcodec")
    parser.addoption("--compare", action="store_true", help="Require and compare against torchcodec 0.17.0")


def _oracle():
    module = importlib.import_module("torchcodec")
    if module.__version__.split("+")[0] != "0.17.0":
        pytest.fail(f"Expected torchcodec 0.17.0, found {module.__version__}")
    return importlib.import_module("torchcodec.decoders")


@pytest.fixture(scope="session")
def backend(request):
    if request.config.getoption("--backend") == "torchcodec":
        return _oracle()
    return importlib.import_module("tensorcodec.decoders")


@pytest.fixture(scope="session")
def oracle(request):
    if not request.config.getoption("--compare"):
        pytest.skip("Differential tests require explicit --compare; CI always enables it")
    return _oracle()


def run_ffmpeg(*args):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True,
        capture_output=True,
    )


@dataclass(frozen=True)
class VideoCase:
    path: Path
    pts: np.ndarray
    durations: np.ndarray
    levels: np.ndarray
    stream_index: int = 0

    @property
    def end(self):
        return float(max(Fraction(str(p)) + Fraction(str(d)) for p, d in zip(self.pts, self.durations)))


def case_from_packets(path, levels, stream_index=0):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_packets", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    packets = sorted(
        (
            p
            for p in json.loads(result.stdout)["packets"]
            if p["stream_index"] == stream_index and "D" not in p["flags"]
        ),
        key=lambda p: float(p["pts_time"]),
    )
    assert len(packets) == len(levels)
    return VideoCase(
        path,
        np.array([float(p["pts_time"]) for p in packets]),
        np.array([float(p.get("duration_time", 0)) for p in packets]),
        np.array(levels, dtype=np.uint8),
        stream_index,
    )


@pytest.fixture(scope="session")
def videos(tmp_path_factory):
    root = tmp_path_factory.mktemp("media")
    levels = np.arange(12, dtype=np.uint8) * 16 + 16
    for i, level in enumerate(levels):
        pixels = np.full((24, 32, 3), level, dtype=np.uint8)
        (root / f"{i:02d}.ppm").write_bytes(b"P6\n32 24\n255\n" + pixels.tobytes())
    # A trailing sentinel gives VFR muxing the final real frame's duration.
    (root / "12.ppm").write_bytes((root / "11.ppm").read_bytes())

    result = {}
    for name, filters, codec, suffix in [
        ("cfr", "", ["-c:v", "ffv1"], "mkv"),
        (
            "vfr",
            "settb=1/1000,setpts=if(lt(N\\,4)\\,N*100\\,400+(N-4)*200)",
            ["-c:v", "libx264", "-qp", "0", "-bf", "0"],
            "mp4",
        ),
        ("offset", "setpts=PTS+2/TB", ["-c:v", "ffv1"], "mkv"),
        (
            "bframes",
            "",
            ["-c:v", "libx264", "-crf", "12", "-bf", "3", "-g", "8", "-x264-params", "b-adapt=0"],
            "mp4",
        ),
    ]:
        path = root / f"{name}.{suffix}"
        if name == "offset":
            run_ffmpeg("-itsoffset", "2", "-i", result["cfr"].path, "-c:v", "copy", path)
            result[name] = case_from_packets(path, levels)
            continue
        args = ["-framerate", "10", "-i", root / "%02d.ppm"]
        if filters:
            args += ["-vf", filters]
        encoded = root / "vfr-with-sentinel.mp4" if name == "vfr" else path
        run_ffmpeg(*args, "-frames:v", "13" if name == "vfr" else "12", "-fps_mode", "vfr", *codec, encoded)
        if name == "vfr":
            # FFmpeg 6 and 7 differ in whether the sentinel is marked DISCARD.
            # Remux exactly the real packets, preserving the last 0.2 s duration.
            run_ffmpeg("-i", encoded, "-c:v", "copy", "-frames:v", "12", path)
        result[name] = case_from_packets(path, levels)
    np.testing.assert_allclose(result["cfr"].pts, np.arange(12) / 10, atol=1e-12)
    np.testing.assert_allclose(result["vfr"].pts, [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
    np.testing.assert_allclose(result["vfr"].durations, [0.1] * 4 + [0.2] * 8, atol=1e-12)
    np.testing.assert_allclose(result["offset"].pts, 2 + np.arange(12) / 10)

    # Put audio first so video stream_index=1 differs from video-list ordinal 0.
    path = root / "multistream.mkv"
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=8000:cl=mono",
        "-i",
        result["cfr"].path,
        "-map",
        "0:a",
        "-map",
        "1:v",
        "-c:a",
        "pcm_s16le",
        "-c:v",
        "copy",
        "-shortest",
        path,
    )
    result["multistream"] = case_from_packets(path, levels, 1)
    return result


@pytest.fixture(scope="session", params=("cfr", "vfr", "offset", "bframes", "multistream"))
def video(request, videos):
    return videos[request.param]


@pytest.fixture(scope="session")
def audio(tmp_path_factory):
    path = tmp_path_factory.mktemp("audio") / "samples.wav"
    rate = 8000
    n = np.arange(rate)
    samples = np.stack(((n * 37) % 30000 - 15000, (n * 53) % 28000 - 14000)).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.T.tobytes())
    return path, samples.astype(np.float32) / 32768, rate


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


@pytest.fixture(scope="session", params=("bt709", "bt2020nc"))
def color_video(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("color") / f"{request.param}.mp4"
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=34x26:rate=10:duration=1.2",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        "-bf",
        "3",
        "-g",
        "8",
        "-colorspace",
        request.param,
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        path,
    )
    return path


@pytest.fixture(scope="session", params=("aac", "mp3", "flac"))
def compressed_audio(request, audio, tmp_path_factory):
    path = (
        tmp_path_factory.mktemp("compressed")
        / {"aac": "samples.m4a", "mp3": "samples.mp3", "flac": "samples.mka"}[request.param]
    )
    run_ffmpeg("-i", audio[0], "-c:a", request.param, path)
    return path
