from __future__ import annotations

import argparse

from person_detector.app import PersonDetectionApp
from person_detector.camera import Camera
from person_detector.detector import PersonDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time person detection using YOLOv8")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera = Camera(index=args.camera_index, width=args.width, height=args.height)
    detector = PersonDetector(model_path=args.model, confidence_threshold=args.confidence)

    app = PersonDetectionApp(camera=camera, detector=detector, track_cpu=not args.no_cpu)
    app.run()


if __name__ == "__main__":
    main()
