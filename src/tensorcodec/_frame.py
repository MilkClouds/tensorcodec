"""NumPy counterparts of TorchCodec's public result objects."""

from dataclasses import dataclass, fields

import numpy as np


def _repr(value):
    lines = [f"{type(value).__name__}:"]
    for field in fields(value):
        item = getattr(value, field.name)
        if field.name == "pixel_format" and item is None:
            continue
        label = "data (shape)" if field.name == "data" else field.name
        lines.append(f"  {label}: {item.shape if field.name == 'data' else item}")
    return "\n".join(lines) + "\n"


@dataclass
class Frame:
    data: np.ndarray
    pts_seconds: float
    duration_seconds: float
    pixel_format: str | None = None

    def __post_init__(self):
        if self.data.ndim != 3:
            raise ValueError(f"data must be 3-dimensional, got {self.data.shape}")
        self.pts_seconds = float(self.pts_seconds)
        self.duration_seconds = float(self.duration_seconds)

    def __iter__(self):
        yield self.data
        yield self.pts_seconds
        yield self.duration_seconds

    def __repr__(self):
        return _repr(self)


@dataclass
class FrameBatch:
    data: np.ndarray
    pts_seconds: np.ndarray
    duration_seconds: np.ndarray
    pixel_format: str | None = None

    def __post_init__(self):
        self.pts_seconds = np.asarray(self.pts_seconds, dtype=np.float64)
        self.duration_seconds = np.asarray(self.duration_seconds, dtype=np.float64)
        if self.data.ndim < 3:
            raise ValueError(f"data must be at least 3-dimensional, got {self.data.shape}")
        leading = self.data.shape[:-3]
        if leading != self.pts_seconds.shape or leading != self.duration_seconds.shape:
            raise ValueError("data, pts_seconds and duration_seconds leading dimensions must match")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key):
        return FrameBatch(self.data[key], self.pts_seconds[key], self.duration_seconds[key], self.pixel_format)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return _repr(self)


@dataclass
class AudioSamples:
    data: np.ndarray
    pts_seconds: float
    duration_seconds: float
    sample_rate: int

    def __post_init__(self):
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2-dimensional, got {self.data.shape}")
        self.pts_seconds = float(self.pts_seconds)
        self.sample_rate = int(self.sample_rate)

    def __iter__(self):
        yield self.data
        yield self.pts_seconds
        yield self.duration_seconds
        yield self.sample_rate

    def __repr__(self):
        return _repr(self)
