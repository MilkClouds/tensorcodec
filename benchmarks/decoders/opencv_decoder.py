"""Benchmark adapter for OpenCV (cv2).

Requires ``opencv-python`` or ``opencv-python-headless`` to be installed.
The benchmark registry in ``__init__.py`` catches the ``ImportError`` if
it is missing.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


class OpenCVDecoder:
    """Wraps :class:`cv2.VideoCapture` for the benchmark protocol."""

    name = "opencv"

    def get_frames_played_at(self, video_path: str, seconds: List[float]) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        try:
            frames = []
            for ts in seconds:
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ret, frame = cap.read()
                if ret:
                    # BGR -> RGB, then HWC -> CHW
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(np.transpose(frame, (2, 0, 1)))
            if not frames:
                return np.empty((0, 3, 0, 0), dtype=np.uint8)
            return np.stack(frames)  # NCHW
        finally:
            cap.release()

    def get_frames_played_in_range(
        self,
        video_path: str,
        start_seconds: float,
        stop_seconds: float,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        if fps is not None:
            raise ValueError("OpenCVDecoder does not support the fps parameter")
        cap = cv2.VideoCapture(video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Check position *after* reading so we use the actual
                # frame's timestamp, not the pre-read seek position.
                pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if pos_sec >= stop_seconds:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(np.transpose(frame, (2, 0, 1)))
            if not frames:
                return np.empty((0, 3, 0, 0), dtype=np.uint8)
            return np.stack(frames)  # NCHW
        finally:
            cap.release()

    def get_video_duration(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        try:
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                return 0.0
            return frame_count / fps
        finally:
            cap.release()
