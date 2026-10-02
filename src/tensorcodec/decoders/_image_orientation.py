"""JPEG, PNG and WebP EXIF orientation."""

import numpy as np


def _tiff_orientation(data):
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return 1
    order = "little" if data[:2] == b"II" else "big"
    if int.from_bytes(data[2:4], order) != 42:
        return 1
    offset = int.from_bytes(data[4:8], order)
    if offset + 2 > len(data):
        return 1
    count = int.from_bytes(data[offset : offset + 2], order)
    for index in range(count):
        entry = offset + 2 + index * 12
        if entry + 12 > len(data):
            return 1
        if int.from_bytes(data[entry : entry + 2], order) == 274:
            if (
                int.from_bytes(data[entry + 2 : entry + 4], order) != 3
                or int.from_bytes(data[entry + 4 : entry + 8], order) != 1
            ):
                return 1
            value = int.from_bytes(data[entry + 8 : entry + 10], order)
            return value if 1 <= value <= 8 else 1
    return 1


def orientation(data, codec):
    if codec == "webp":
        offset = 12
        while offset + 8 <= len(data):
            size = int.from_bytes(data[offset + 4 : offset + 8], "little")
            if offset + 8 + size > len(data):
                break
            if data[offset : offset + 4] == b"EXIF":
                payload = data[offset + 8 : offset + 8 + size]
                return _tiff_orientation(payload.removeprefix(b"Exif\0\0"))
            offset += 8 + size + (size & 1)
    elif codec == "png":
        offset = 8
        while offset + 12 <= len(data):
            size = int.from_bytes(data[offset : offset + 4], "big")
            if offset + 12 + size > len(data):
                break
            if data[offset + 4 : offset + 8] == b"eXIf":
                return _tiff_orientation(data[offset + 8 : offset + 8 + size])
            offset += size + 12
    elif codec == "jpeg":
        offset = 2
        while offset < len(data) and data[offset] == 0xFF:
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data) or data[offset] in (0xDA, 0xD9):
                break
            marker, offset = data[offset], offset + 1
            size = int.from_bytes(data[offset : offset + 2], "big")
            if size < 2 or offset + size > len(data):
                break
            payload = data[offset + 2 : offset + size]
            if marker == 0xE1 and payload.startswith(b"Exif\0\0"):
                return _tiff_orientation(payload[6:])
            offset += size
    return 1


def apply_orientation(images, value):
    # NHWC throughout; the public wrapper moves channels only after orientation.
    if value == 2:
        return images[:, :, ::-1]
    if value == 3:
        return images[:, ::-1, ::-1]
    if value == 4:
        return images[:, ::-1]
    if value == 5:
        return images.swapaxes(1, 2)
    if value == 6:
        return np.rot90(images, -1, axes=(1, 2))
    if value == 7:
        return images.swapaxes(1, 2)[:, ::-1, ::-1]
    if value == 8:
        return np.rot90(images, 1, axes=(1, 2))
    return images
