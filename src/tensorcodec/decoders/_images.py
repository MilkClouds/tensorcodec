"""CPU image decoding with TorchCodec's function API and NumPy output."""

from enum import Enum
from pathlib import Path

import numpy as np

from tensorcodec._native import decode_image as _decode
from tensorcodec.decoders._image_orientation import apply_orientation, orientation


class ImageReadMode(Enum):
    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4
    RGBA = RGB_ALPHA


def _mode(value):
    if isinstance(value, ImageReadMode):
        return value
    if isinstance(value, str):
        try:
            return ImageReadMode[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid image mode: {value!r}") from None
    raise TypeError("mode must be a string or ImageReadMode")


def _dtype(value):
    if isinstance(value, str) and value == "auto":
        return value
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError):
        raise ValueError("output_dtype must be uint8, uint16 or 'auto'") from None
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("output_dtype must be uint8, uint16 or 'auto'")
    return dtype


def _bytes(source):
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8 or source.ndim != 1:
            raise ValueError("encoded image arrays must be one-dimensional uint8")
        return source.tobytes()
    raise TypeError("source must be a path, bytes or a one-dimensional uint8 NumPy array")


def _format(data):
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] == b"ftyp":
        size = int.from_bytes(data[:4], "big")
        brands = [data[8:12], *[data[i : i + 4] for i in range(16, min(size, len(data)), 4)]]
        if any(b in (b"avif", b"avis") for b in brands):
            return "avif"
        if any(b in (b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1") for b in brands):
            return "heic"
    raise ValueError("Unsupported or unrecognized image format")


def _png_channels(data):
    channels = None
    offset = 8
    while offset + 12 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        if offset + size + 12 > len(data):
            raise RuntimeError("truncated PNG chunk")
        if kind == b"IHDR" and size == 13:
            channels = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}.get(data[offset + 17])
        elif kind == b"tRNS":
            channels = 2 if channels == 1 else 4
        elif kind == b"acTL":
            raise NotImplementedError("animated PNG decoding is unsupported")
        elif kind == b"IDAT":
            break
        offset += size + 12
    if channels is None:
        raise RuntimeError("PNG has no valid IHDR")
    return channels


def _color(images, mode, codec):
    channels = images.shape[-1]
    if mode is ImageReadMode.UNCHANGED:
        return images
    if channels == mode.value:
        return images
    gray_source = channels < 3
    color = images[..., :1] if gray_source else images[..., :3]
    alpha = images[..., -1:] if channels in (2, 4) else np.full_like(images[..., :1], np.iinfo(images.dtype).max)
    if mode in (ImageReadMode.GRAY, ImageReadMode.GRAY_ALPHA):
        if not gray_source:
            if codec == "png":
                # TorchCodec requests libpng coefficients 0.2989 and 0.587.
                color = (
                    (color.astype(np.uint64) * np.array([9794, 19234, 3740], np.uint64)).sum(axis=-1, keepdims=True)
                    >> 15
                ).astype(images.dtype)
            else:
                color = (
                    np.rint(
                        (color.astype(np.float32) * np.array([0.2989, 0.587, 0.114], np.float32)).sum(
                            axis=-1, keepdims=True
                        )
                    )
                    .clip(0, np.iinfo(images.dtype).max)
                    .astype(images.dtype)
                )
    elif gray_source:
        color = np.repeat(color, 3, axis=-1)
    return (
        np.concatenate((color, alpha), axis=-1)
        if mode in (ImageReadMode.GRAY_ALPHA, ImageReadMode.RGB_ALPHA)
        else color
    )


def _image(source, codec, mode, output_dtype, threads=1):
    mode, dtype = _mode(mode), _dtype(output_dtype)
    data = _bytes(source)
    found = _format(data)
    if codec is not None and found != codec:
        raise RuntimeError(f"expected {codec}, got {found}")
    codec = found
    orient = orientation(data, codec)
    animated = False
    if codec == "png":
        _png_channels(data)  # Reject APNG rather than silently returning its first frame.
    if codec == "heic":
        from tensorcodec.decoders._image_libraries import heic

        images = heic(data, high_depth=dtype != np.uint8)
    else:
        depth = 0 if dtype == "auto" else np.dtype(dtype).itemsize * 8
        images, native_orientation, animated = _decode(
            data, ("jpeg", "png", "webp", "gif", "avif").index(codec), mode.value, depth, threads
        )
        if codec == "avif":
            orient = native_orientation
    images = _color(images, mode, codec)
    if dtype != "auto" and images.dtype != dtype:
        if dtype == np.uint16:
            images = images.astype(np.uint16) * 257
        else:
            images = np.rint(images.astype(np.float32) / 257).clip(0, 255).astype(np.uint8)
    images = apply_orientation(images, orient).transpose(0, 3, 1, 2)
    keep_batch = codec == "webp" and animated
    return images[0] if len(images) == 1 and not keep_batch else images


def decode_image(source, *, mode="RGB", output_dtype=np.uint8):
    """Detect encoded content and return a CHW image or NCHW animation on CPU.

    Sources are paths, bytes or 1-D uint8 arrays. Modes: UNCHANGED, GRAY,
    GRAY_ALPHA, RGB, RGB_ALPHA (case-insensitive strings or ImageReadMode).
    output_dtype is uint8, uint16 or 'auto'; integer conversion scales the range.
    HEIC requires system libheif. Other image codecs are bundled in wheels.
    Animated PNG is unsupported.
    """
    return _image(source, None, mode, output_dtype)


def decode_jpeg(source, *, mode="RGB", output_dtype=np.uint8, device="cpu"):
    """Decode a JPEG to CHW, or a list/tuple of sources to a list of CHW arrays.

    Only CPU decoding is supported. Batches may have different image dimensions.
    """
    if str(device) != "cpu":
        raise ValueError("only CPU image decoding is supported")
    if isinstance(source, (list, tuple)):
        _mode(mode)
        _dtype(output_dtype)
        return [_image(item, "jpeg", mode, output_dtype) for item in source]
    return _image(source, "jpeg", mode, output_dtype)


def decode_png(source, *, mode="RGB", output_dtype=np.uint8):
    """Decode a PNG to CHW; 'auto' preserves native 8/16-bit sample precision."""
    return _image(source, "png", mode, output_dtype)


def decode_webp(source, *, mode="RGB", output_dtype=np.uint8):
    """Decode WebP to CHW/NCHW. Animations are composited with libwebpdemux."""
    return _image(source, "webp", mode, output_dtype)


def decode_gif(source, *, mode="RGB", output_dtype=np.uint8):
    """Decode GIF to CHW for one frame or NCHW for multiple frames."""
    return _image(source, "gif", mode, output_dtype)


def decode_avif(source, *, mode="RGB", output_dtype=np.uint8, num_threads=1):
    """Decode AVIF to CHW/NCHW; num_threads controls libavif workers."""
    if not isinstance(num_threads, int) or isinstance(num_threads, bool) or num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    return _image(source, "avif", mode, output_dtype, num_threads)


def decode_heic(source, *, mode="RGB", output_dtype=np.uint8):
    """Decode HEIC/HEIF to CHW/NCHW using optional system libheif."""
    return _image(source, "heic", mode, output_dtype)
