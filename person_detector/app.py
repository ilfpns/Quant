from __future__ import annotations

import cv2

from person_detector.camera import Camera
from person_detector.detector import PersonDetector
from person_detector.visualizer import draw_detections

WINDOW_NAME = "Person Detection"
QUIT_KEY = "q"


class PersonDetectionApp:
    def __init__(self, camera: Camera, detector: PersonDetector) -> None:
        self._camera = camera
        self._detector = detector

    def run(self) -> None:
        with self._camera as camera:
            while True:
                frame = camera.read()
                if frame is None:
                    break

                detections = self._detector.detect(frame)
                annotated = draw_detections(frame, detections)

                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
                    break

        cv2.destroyAllWindows()
