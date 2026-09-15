from __future__ import annotations

import argparse
import time

from person_detector.camera import Camera
from person_detector.detector import PersonDetector
from person_detector.metrics import PerformanceMonitor, print_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 person detection performance")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--no-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera = Camera(index=args.camera_index, width=args.width, height=args.height)
    detector = PersonDetector(model_path=args.model, confidence_threshold=args.confidence)
    monitor = PerformanceMonitor(model_path=args.model, track_cpu=not args.no_cpu)

    with camera:
        monitor.start()
        deadline = time.perf_counter() + args.duration
        try:
            while time.perf_counter() < deadline:
                frame = camera.read()
                if frame is None:
                    continue

                inference_start = time.perf_counter()
                detector.detect(frame)
                inference_seconds = time.perf_counter() - inference_start

                monitor.record_frame(inference_seconds)
        except KeyboardInterrupt:
            pass

    print_report(monitor.result(), monitor.cpu_tracking_enabled)


if __name__ == "__main__":
    main()
