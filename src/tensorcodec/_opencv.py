"""Lazy access to the user's OpenCV installation."""


def opencv():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Image codecs require OpenCV >= 4.13; install tensorcodec[images] "
            "or use an existing compatible cv2 installation"
        ) from exc
    if tuple(int(part) for part in cv2.__version__.split(".")[:2]) < (4, 13):
        raise ImportError("Image codecs require OpenCV >= 4.13")
    return cv2
