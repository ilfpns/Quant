from __future__ import annotations

import time

import cv2

from person_detector.camera import Camera
from person_detector.detector import PersonDetector
from person_detector.metrics import PerformanceMonitor, print_report
from person_detector.visualizer import draw_detections

WINDOW_NAME = "Person Detection"
QUIT_KEY = "q"
REPORT_KEY = "c"
MAX_CONSECUTIVE_READ_FAILURES = 30


class PersonDetectionApp:
    def __init__(self, camera: Camera, detector: PersonDetector, track_cpu: bool = True) -> None:
        self._camera = camera
        self._detector = detector
        self._monitor = PerformanceMonitor(model_path=detector.model_path, track_cpu=track_cpu)

    def run(self) -> None:
        consecutive_failures = 0
        with self._camera as camera:
            self._monitor.start()
            while True:
                frame = camera.read()
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                        break
                    continue
                consecutive_failures = 0

                inference_start = time.perf_counter()
                detections = self._detector.detect(frame)
                self._monitor.record_frame(time.perf_counter() - inference_start)

                annotated = draw_detections(frame, detections)

                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(QUIT_KEY):
                    break
                if key == ord(REPORT_KEY):
                    print_report(self._monitor.result(), self._monitor.cpu_tracking_enabled)

        cv2.destroyAllWindows()
