"""Optional system-library bindings for formats FFmpeg 7 cannot decode completely."""

import ctypes as c
from ctypes.util import find_library
from functools import lru_cache

import numpy as np


@lru_cache
def _library(name):
    path = find_library(name)
    if path is None:
        raise ImportError(f"image decoding requires the system library lib{name}")
    try:
        return c.CDLL(path)
    except OSError as exc:
        raise ImportError(f"cannot load lib{name}: {exc}") from exc


def _function(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype, function.argtypes = result, arguments
    return function


class _WebPData(c.Structure):
    _fields_ = [("bytes", c.c_void_p), ("size", c.c_size_t)]


class _WebPInfo(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("width", "height", "loop", "background", "frames")] + [
        ("padding", c.c_uint32 * 4)
    ]


def webp_animation(data):
    library = _library("webpdemux")
    new = _function(library, "WebPAnimDecoderNewInternal", c.c_void_p, c.POINTER(_WebPData), c.c_void_p, c.c_int)
    info = _function(library, "WebPAnimDecoderGetInfo", c.c_int, c.c_void_p, c.POINTER(_WebPInfo))
    more = _function(library, "WebPAnimDecoderHasMoreFrames", c.c_int, c.c_void_p)
    next_frame = _function(
        library, "WebPAnimDecoderGetNext", c.c_int, c.c_void_p, c.POINTER(c.POINTER(c.c_uint8)), c.POINTER(c.c_int)
    )
    delete = _function(library, "WebPAnimDecoderDelete", None, c.c_void_p)
    buffer = c.create_string_buffer(data)
    source = _WebPData(c.cast(buffer, c.c_void_p), len(data))
    decoder = new(c.byref(source), None, 0x0107)  # Stable libwebp demux ABI; default output is RGBA.
    if not decoder:
        raise RuntimeError("cannot open animated WebP")
    try:
        metadata = _WebPInfo()
        if not info(decoder, c.byref(metadata)) or not metadata.width or not metadata.height:
            raise RuntimeError("invalid WebP canvas")
        frames = []
        while more(decoder):
            pixels, timestamp = c.POINTER(c.c_uint8)(), c.c_int()
            if not next_frame(decoder, c.byref(pixels), c.byref(timestamp)) or not pixels:
                raise RuntimeError("cannot decode WebP frame")
            frame = np.ctypeslib.as_array(pixels, shape=(metadata.height * metadata.width * 4,))
            frames.append(frame.reshape(metadata.height, metadata.width, 4).copy())
        if len(frames) != metadata.frames or not frames:
            raise RuntimeError("WebP frame count differs from its header")
        return np.stack(frames)
    finally:
        delete(decoder)


class _HeifError(c.Structure):
    _fields_ = [("code", c.c_int), ("subcode", c.c_int), ("message", c.c_char_p)]


def _check(error):
    if error.code:
        raise RuntimeError(error.message.decode("utf-8", errors="replace") if error.message else "libheif error")


def heic(data, high_depth):
    library = _library("heif")
    pointer = c.c_void_p
    allocate = _function(library, "heif_context_alloc", pointer)
    free = _function(library, "heif_context_free", None, pointer)
    read = _function(
        library, "heif_context_read_from_memory_without_copy", _HeifError, pointer, pointer, c.c_size_t, pointer
    )
    count = _function(library, "heif_context_get_number_of_top_level_images", c.c_int, pointer)
    ids = _function(
        library, "heif_context_get_list_of_top_level_image_IDs", c.c_int, pointer, c.POINTER(c.c_uint32), c.c_int
    )
    get_handle = _function(
        library, "heif_context_get_image_handle", _HeifError, pointer, c.c_uint32, c.POINTER(pointer)
    )
    release_handle = _function(library, "heif_image_handle_release", None, pointer)
    depth = _function(library, "heif_image_handle_get_luma_bits_per_pixel", c.c_int, pointer)
    alpha = _function(library, "heif_image_handle_has_alpha_channel", c.c_int, pointer)
    decode = _function(
        library, "heif_decode_image", _HeifError, pointer, c.POINTER(pointer), c.c_int, c.c_int, pointer
    )
    release_image = _function(library, "heif_image_release", None, pointer)
    width = _function(library, "heif_image_get_width", c.c_int, pointer, c.c_int)
    height = _function(library, "heif_image_get_height", c.c_int, pointer, c.c_int)
    plane = _function(
        library, "heif_image_get_plane_readonly", c.POINTER(c.c_uint8), pointer, c.c_int, c.POINTER(c.c_int)
    )
    context = allocate()
    if not context:
        raise MemoryError("cannot allocate HEIC context")
    buffer = c.create_string_buffer(data)
    try:
        _check(read(context, buffer, len(data), None))
        size = count(context)
        if size <= 0:
            raise RuntimeError("HEIC contains no images")
        items = (c.c_uint32 * size)()
        if ids(context, items, size) != size:
            raise RuntimeError("incomplete HEIC image list")
        frames, layout = [], None
        for item in items:
            handle, image = pointer(), pointer()
            try:
                _check(get_handle(context, item, c.byref(handle)))
                bits, channels = depth(handle), 4 if alpha(handle) else 3
                if bits not in (8, 10, 12, 16):
                    raise RuntimeError(f"unsupported HEIC bit depth: {bits}")
                high = bits > 8 and high_depth
                chroma = (14 if high else 10) + (channels == 4)  # Interleaved RGB(A), little-endian for 16-bit.
                _check(decode(handle, c.byref(image), 1, chroma, None))
                w, h, stride = width(image, 10), height(image, 10), c.c_int()
                pixels = plane(image, 10, c.byref(stride))
                row_size = w * channels * (2 if high else 1)
                if not pixels or w <= 0 or h <= 0 or stride.value < row_size:
                    raise RuntimeError("invalid HEIC image plane")
                current = (h, w, channels, bits)
                if layout is not None and current != layout:
                    raise RuntimeError("HEIC images have different dimensions, channels or bit depths")
                layout = current
                rows = np.ctypeslib.as_array(pixels, shape=(h * stride.value,)).reshape(h, stride.value)
                frame = rows[:, :row_size].copy().view("<u2" if high else np.uint8).reshape(h, w, channels)
                if high:
                    shift = 16 - bits
                    frame = (frame << shift) | (frame >> (bits - shift))
                frames.append(frame)
            finally:
                if image:
                    release_image(image)
                if handle:
                    release_handle(handle)
        return np.stack(frames)
    finally:
        free(context)
