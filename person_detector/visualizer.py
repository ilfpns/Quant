from __future__ import annotations

import cv2
import numpy as np

from person_detector.detector import Detection

BOX_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()

    for detection in detections:
        cv2.rectangle(
            annotated,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            BOX_COLOR,
            2,
        )
        label = f"person {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (detection.x1, max(detection.y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2,
        )

    cv2.putText(
        annotated,
        f"persons: {len(detections)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
    )
    return annotated
