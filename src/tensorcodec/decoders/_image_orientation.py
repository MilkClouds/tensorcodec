"""JPEG, PNG and WebP EXIF orientation."""

import numpy as np


def _boxes(data, start=0, end=None):
    end = len(data) if end is None else end
    while start < end:
        if start + 8 > end:
            raise RuntimeError("truncated AVIF box")
        size, kind = int.from_bytes(data[start : start + 4], "big"), data[start + 4 : start + 8]
        header = 8
        if size == 1:
            if start + 16 > end:
                raise RuntimeError("truncated extended AVIF box")
            size, header = int.from_bytes(data[start + 8 : start + 16], "big"), 16
        elif size == 0:
            size = end - start
        if size < header or start + size > end:
            raise RuntimeError("invalid AVIF box size")
        yield kind, start + header, start + size
        start += size


def _avif_orientation(data):
    properties, associations, primary = [], [], None
    for kind, begin, end in _boxes(data):
        if kind != b"meta":
            continue
        for child, lo, hi in _boxes(data, begin + 4, end):
            if child == b"pitm":
                width = 2 if data[lo] == 0 else 4
                if lo + 4 + width > hi:
                    raise RuntimeError("truncated AVIF primary item")
                primary = int.from_bytes(data[lo + 4 : lo + 4 + width], "big")
            if child != b"iprp":
                continue
            for prop, p, q in _boxes(data, lo, hi):
                if prop == b"ipco":
                    properties = list(_boxes(data, p, q))
                elif prop == b"ipma":
                    if p + 8 > q:
                        raise RuntimeError("truncated AVIF property associations")
                    width = 2 if data[p] == 0 else 4
                    entry_width = 2 if int.from_bytes(data[p + 1 : p + 4], "big") & 1 else 1
                    count, cursor = int.from_bytes(data[p + 4 : p + 8], "big"), p + 8
                    for _ in range(count):
                        if cursor + width + 1 > q:
                            raise RuntimeError("truncated AVIF item association")
                        item = int.from_bytes(data[cursor : cursor + width], "big")
                        n = data[cursor + width]
                        cursor += width + 1
                        if cursor + n * entry_width > q:
                            raise RuntimeError("truncated AVIF property indices")
                        indices = [
                            int.from_bytes(data[i : i + entry_width], "big") & ((1 << (entry_width * 8 - 1)) - 1)
                            for i in range(cursor, cursor + n * entry_width, entry_width)
                        ]
                        associations.append((item, indices))
                        cursor += n * entry_width
    angle, axis = 0, None
    for item, indices in associations:
        if item != primary:
            continue
        for index in indices:
            if index == 0:
                continue
            if index > len(properties):
                raise RuntimeError("invalid AVIF property index")
            kind, start, end = properties[index - 1]
            if kind in (b"irot", b"imir"):
                if start == end:
                    raise RuntimeError("empty AVIF orientation property")
                if kind == b"irot":
                    angle = data[start] & 3
                else:
                    axis = data[start] & 1
    # ISO/IEC 23008-12 item rotation/mirror mapped to TIFF's eight orientations.
    return ((1, 4, 2), (8, 5, 7), (3, 2, 4), (6, 7, 5))[angle][0 if axis is None else axis + 1]


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
    if codec == "avif":
        return _avif_orientation(data)
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
