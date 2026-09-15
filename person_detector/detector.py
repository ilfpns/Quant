from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from person_detector.quantization import dynamic_quantize

PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        quantize: bool = False,
    ) -> None:
        self.model_path = model_path
        self.is_quantized = quantize
        self._model = YOLO(model_path)
        if quantize:
            self._model.model = dynamic_quantize(self._model.model)
        self._confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self._confidence_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            detections.append(Detection(int(x1), int(y1), int(x2), int(y2), confidence))
        return detections
