from __future__ import annotations

import time

import cv2
import numpy as np

WARMUP_TIMEOUT_SECONDS = 3.0


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._capture = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera at index {self._index}")
        self._warm_up()

    def _warm_up(self) -> None:
        deadline = time.monotonic() + WARMUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            ok, _ = self._capture.read()
            if ok:
                return
        raise RuntimeError(f"Camera at index {self._index} opened but never produced a frame")

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            raise RuntimeError("Camera is not open. Call open() first.")
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
