"""Image API contracts from known PNG samples and independent FFmpeg encodings."""

import shutil
import struct
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.utils import as_numpy, run_ffmpeg


def png_bytes(pixels):
    height, width, channels = pixels.shape
    depth = pixels.dtype.itemsize * 8
    color = {1: 0, 2: 4, 3: 2, 4: 6}[channels]

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = pixels.astype(">u2" if depth == 16 else np.uint8).tobytes()
    stride = width * channels * (depth // 8)
    scanlines = b"".join(b"\0" + raw[i : i + stride] for i in range(0, len(raw), stride))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def dtype_for(backend, name):
    if backend.__name__.startswith("torchcodec"):
        import torch

        return getattr(torch, name)
    return getattr(np, name)


@pytest.fixture(scope="module")
def encoded_images(tmp_path_factory):
    root = tmp_path_factory.mktemp("images")
    pixels = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    (root / "source.png").write_bytes(png_bytes(pixels))
    for codec, options in {
        "jpeg": ["-c:v", "mjpeg", "-q:v", "1", "-pix_fmt", "yuvj444p"],
        "gif": [],
        "avif": ["-c:v", "libaom-av1", "-still-picture", "1", "-crf", "0", "-cpu-used", "8"],
    }.items():
        run_ffmpeg("-i", root / "source.png", "-frames:v", 1, *options, "-threads", 1, root / f"image.{codec}")
    Image.fromarray(pixels).save(root / "image.webp", lossless=True)
    (root / "second.png").write_bytes(png_bytes(255 - pixels))
    run_ffmpeg(
        "-framerate",
        2,
        "-pattern_type",
        "glob",
        "-i",
        str(root / "s*.png"),
        "-threads",
        1,
        root / "animated.gif",
    )
    return root


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("depth", [8, 16])
def test_png_native_samples(backend, channels, depth):
    dtype = np.uint8 if depth == 8 else np.uint16
    pixels = np.arange(5 * 7 * channels, dtype=dtype).reshape(5, 7, channels)
    if depth == 16:
        pixels = pixels * 311
    decoded = backend.decode_png(png_bytes(pixels), mode="UNCHANGED", output_dtype="auto")
    np.testing.assert_array_equal(as_numpy(decoded), pixels.transpose(2, 0, 1))
    assert as_numpy(decoded).dtype == dtype


@pytest.mark.parametrize("mode,channels", [("RGB", 3), ("GRAY", 1), ("RGB_ALPHA", 4), ("GRAY_ALPHA", 2)])
def test_png_color_modes(backend, mode, channels):
    pixels = np.array([[[255, 0, 0, 17], [0, 255, 0, 93], [0, 0, 255, 201]]], np.uint8)
    result = as_numpy(backend.decode_image(png_bytes(pixels), mode=mode.lower()))
    assert result.shape == (channels, 1, 3)
    if "ALPHA" in mode:
        np.testing.assert_array_equal(result[-1], pixels[..., 3])
    if mode.startswith("RGB"):
        np.testing.assert_array_equal(result[:3], pixels[..., :3].transpose(2, 0, 1))
    else:
        np.testing.assert_array_equal(result[0], [[76, 149, 29]])


def test_output_dtype_scales_values(backend):
    pixels = np.array([[[0], [1], [128], [255]]], np.uint8)
    result = backend.decode_png(png_bytes(pixels), mode="GRAY", output_dtype=dtype_for(backend, "uint16"))
    np.testing.assert_array_equal(as_numpy(result), pixels.transpose(2, 0, 1).astype(np.uint16) * 257)
    pixels = np.array([[[0], [128], [129], [32768], [65535]]], np.uint16)
    result = backend.decode_png(png_bytes(pixels), mode="GRAY")
    np.testing.assert_array_equal(as_numpy(result), np.rint(pixels.transpose(2, 0, 1) / 257).astype(np.uint8))


@pytest.mark.parametrize("kind", ["bytes", "bytearray", "path", "str", "array"])
def test_image_sources_and_content_detection(backend, tmp_path, kind):
    pixels = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    data = png_bytes(pixels)
    path = tmp_path / "misleading.jpg"
    path.write_bytes(data)
    sources = {"bytes": data, "bytearray": bytearray(data), "path": path, "str": str(path)}
    if kind == "array":
        if backend.__name__.startswith("torchcodec"):
            import torch

            source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        else:
            source = np.frombuffer(data, dtype=np.uint8)
    else:
        source = sources[kind]
    np.testing.assert_array_equal(as_numpy(backend.decode_image(source)), pixels.transpose(2, 0, 1))


@pytest.mark.parametrize("codec", ["jpeg", "webp", "gif", "avif"])
def test_format_functions_and_dispatch(backend, encoded_images, codec):
    path = encoded_images / f"image.{codec}"
    direct = as_numpy(getattr(backend, f"decode_{codec}")(path))
    assert direct.shape == (3, 16, 24)
    assert direct.dtype == np.uint8
    np.testing.assert_array_equal(direct, as_numpy(backend.decode_image(path.read_bytes())))


def test_jpeg_batch(backend, encoded_images):
    source = encoded_images / "image.jpeg"
    images = backend.decode_jpeg([source, source.read_bytes()])
    assert isinstance(images, list)
    assert len(images) == 2
    np.testing.assert_array_equal(as_numpy(images[0]), as_numpy(images[1]))
    images[0][...] = 0
    assert as_numpy(images[1]).any()
    assert backend.decode_jpeg([]) == []


def test_gif_animation(backend, encoded_images):
    frames = as_numpy(backend.decode_gif(encoded_images / "animated.gif"))
    assert frames.shape == (2, 3, 16, 24)
    assert not np.array_equal(frames[0], frames[1])


@pytest.mark.parametrize("mode", ["RGB", "UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB_ALPHA"])
@pytest.mark.parametrize("codec", ["jpeg", "png", "webp", "gif", "avif"])
def test_image_differential(oracle, encoded_images, codec, mode):
    import tensorcodec.decoders as actual

    path = encoded_images / ("source.png" if codec == "png" else f"image.{codec}")
    got = getattr(actual, f"decode_{codec}")(path, mode=mode)
    expected = as_numpy(getattr(oracle, f"decode_{codec}")(path, mode=mode))
    assert got.shape == expected.shape
    np.testing.assert_allclose(got.astype(np.int32), expected.astype(np.int32), atol=2, rtol=0)


def test_image_errors_and_cpu_boundary():
    from tensorcodec.decoders import decode_avif, decode_image, decode_jpeg, decode_png

    data = png_bytes(np.zeros((2, 3, 3), np.uint8))
    with pytest.raises(ValueError, match="mode"):
        decode_png(data, mode="BGR")
    with pytest.raises(ValueError, match="output_dtype"):
        decode_png(data, output_dtype=np.float32)
    with pytest.raises(ValueError, match="one-dimensional"):
        decode_png(np.zeros((2, 3), np.uint8))
    with pytest.raises(TypeError, match="source"):
        decode_image(object())
    with pytest.raises(ValueError, match="CPU"):
        decode_jpeg(b"", device="cuda")
    with pytest.raises(RuntimeError, match="expected jpeg"):
        decode_jpeg(data)
    with pytest.raises(ValueError, match="unrecognized"):
        decode_image(b"not an image")
    with pytest.raises(ValueError, match="num_threads"):
        decode_avif(b"", num_threads=0)
    with pytest.raises(FileNotFoundError):
        decode_image(Path("/nonexistent/image.png"))


def test_optional_libraries_report_missing_dependencies(monkeypatch):
    from tensorcodec.decoders import _image_libraries, decode_image

    monkeypatch.setattr(_image_libraries, "find_library", lambda name: None)
    _image_libraries._library.cache_clear()
    heic = struct.pack(">I", 24) + b"ftypheic" + b"\0" * 4 + b"mif1heic"
    with pytest.raises(ImportError, match="libheif"):
        decode_image(heic)
    animated = b"RIFF" + struct.pack("<I", 12) + b"WEBPANIM" + b"\0" * 4
    with pytest.raises(ImportError, match="libwebpdemux"):
        decode_image(animated)


@pytest.fixture(scope="module")
def optional_images(tmp_path_factory):
    root = tmp_path_factory.mktemp("optional-images")
    pixels = np.zeros((16, 24, 4), np.uint8)
    pixels[..., :3] = [30, 60, 90]
    pixels[..., 3] = 255
    pixels[:8, :12, 3] = 0
    image = Image.fromarray(pixels)
    second = image.copy()
    second.paste((120, 50, 10, 128), (8, 4, 16, 12))
    image.save(root / "animation.webp", save_all=True, append_images=[second], lossless=True, duration=100)
    fixtures = Path(__file__).parent / "resources/images"
    for source, target in [
        ("rgba.heic", "alpha.heic"),
        ("animated.heic", "multi.heic"),
        ("gradient_10bit.heic", "high.heic"),
    ]:
        shutil.copyfile(fixtures / source, root / target)
    return root


@pytest.mark.parametrize("file", ["animation.webp", "alpha.heic", "multi.heic", "high.heic"])
@pytest.mark.parametrize("mode", ["RGB", "UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB_ALPHA"])
def test_optional_image_differential(oracle, optional_images, file, mode):
    from tensorcodec.decoders import _image_libraries, decode_image

    library = "webpdemux" if file.endswith("webp") else "heif"
    try:
        _image_libraries._library(library)
    except ImportError:
        pytest.skip(f"optional system library {library} unavailable")
    path = optional_images / file
    got = decode_image(path, mode=mode)
    expected = as_numpy(oracle.decode_image(path, mode=mode))
    assert got.shape == expected.shape
    np.testing.assert_allclose(got.astype(np.int32), expected.astype(np.int32), atol=2, rtol=0)


@pytest.mark.parametrize("dtype", ["auto", "uint16"])
def test_heic_high_depth(oracle, optional_images, dtype):
    from tensorcodec.decoders import _image_libraries, decode_heic

    try:
        _image_libraries._library("heif")
    except ImportError:
        pytest.skip("optional system libheif unavailable")
    actual = decode_heic(optional_images / "high.heic", output_dtype=dtype)
    expected = as_numpy(
        oracle.decode_heic(
            optional_images / "high.heic", output_dtype="auto" if dtype == "auto" else dtype_for(oracle, dtype)
        )
    )
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)


def test_transparent_gif_matches_oracle(oracle, tmp_path):
    from tensorcodec.decoders import decode_gif

    image = Image.new("P", (16, 16))
    image.putpalette([90, 30, 10, 10, 150, 30] + [0] * 762)
    image.paste(1, (4, 4, 12, 12))
    path = tmp_path / "transparent.gif"
    image.save(path, transparency=0, background=0)
    for mode in ("RGB", "UNCHANGED", "GRAY_ALPHA"):
        actual = decode_gif(path, mode=mode)
        expected = as_numpy(oracle.decode_gif(path, mode=mode))
        assert actual.shape == expected.shape
        if actual.shape[0] in (2, 4):
            np.testing.assert_array_equal(actual[-1], expected[-1])
            opaque = actual[-1] != 0
            np.testing.assert_allclose(actual[:-1, opaque], expected[:-1, opaque], atol=1, rtol=0)
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("disposal", [1, 2, 3])
def test_gif_disposal(oracle, tmp_path, disposal):
    from tensorcodec.decoders import decode_gif

    frames = []
    for index in range(3):
        image = Image.new("P", (16, 16))
        image.putpalette([90, 30, 10, 10, 150, 30] + [0] * 762)
        image.paste(1, (index * 4, index * 4, index * 4 + 4, index * 4 + 4))
        frames.append(image)
    path = tmp_path / "disposal.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], transparency=0, background=0, disposal=disposal)
    for mode in ("RGB", "RGB_ALPHA"):
        actual, expected = decode_gif(path, mode=mode), as_numpy(oracle.decode_gif(path, mode=mode))
        if mode == "RGB_ALPHA":
            np.testing.assert_array_equal(actual[:, 3], expected[:, 3])
            mask = actual[:, 3] != 0
            actual, expected = actual.transpose(0, 2, 3, 1)[mask], expected.transpose(0, 2, 3, 1)[mask]
        np.testing.assert_array_equal(actual, expected)


def test_high_depth_avif(oracle, tmp_path):
    from tensorcodec.decoders import decode_avif

    pixels = np.arange(16 * 24 * 3, dtype=np.uint16).reshape(16, 24, 3) * 53
    source, path = tmp_path / "high.png", tmp_path / "high.avif"
    source.write_bytes(png_bytes(pixels))
    run_ffmpeg(
        "-i",
        source,
        "-frames:v",
        1,
        "-c:v",
        "libaom-av1",
        "-pix_fmt",
        "yuv444p10le",
        "-still-picture",
        1,
        "-crf",
        0,
        "-cpu-used",
        8,
        "-threads",
        1,
        path,
    )
    actual = decode_avif(path, output_dtype="auto")
    expected = as_numpy(oracle.decode_avif(path, output_dtype="auto"))
    assert actual.dtype == np.uint16
    # FFmpeg and libavif use different high-depth YUV conversion/scaling paths.
    # Bound the difference to one 8-bit-equivalent level, not native-sample equality.
    np.testing.assert_allclose(actual.astype(np.int32), expected.astype(np.int32), atol=257, rtol=0)


def test_corrupt_png_raises():
    from tensorcodec.decoders import decode_png

    data = png_bytes(np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(RuntimeError):
        decode_png(data[:45])


def test_cmyk_jpeg_modes(oracle):
    from tensorcodec.decoders import decode_jpeg

    image = Image.new("CMYK", (8, 8), (30, 50, 70, 90))
    encoded = BytesIO()
    image.save(encoded, format="JPEG")
    data = encoded.getvalue()
    for mode in ("RGB", "GRAY", "RGB_ALPHA", "GRAY_ALPHA"):
        actual, expected = decode_jpeg(data, mode=mode), as_numpy(oracle.decode_jpeg(data, mode=mode))
        np.testing.assert_allclose(actual.astype(int), expected.astype(int), atol=2, rtol=0)
        if "ALPHA" in mode:
            assert (actual[-1] == 255).all()
    with pytest.raises(NotImplementedError, match="CMYK"):
        decode_jpeg(data, mode="UNCHANGED")
