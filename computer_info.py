from __future__ import annotations

import argparse
import time

from person_detector.camera import Camera
from person_detector.detector import PersonDetector
from person_detector.metrics import PerformanceMonitor, append_report_to_log, print_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 person detection performance")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--no-cpu", action="store_true")
    parser.add_argument("--quantize", action="store_true", help="apply PyTorch dynamic quantization")
    parser.add_argument("--label", type=str, default=None, help="row label for the log file")
    parser.add_argument("--log-file", type=str, default="benchmark_log.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera = Camera(index=args.camera_index, width=args.width, height=args.height)
    detector = PersonDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        quantize=args.quantize,
    )
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

    result = monitor.result()
    print_report(result, monitor.cpu_tracking_enabled)

    label = args.label or ("ptq_dynamic" if args.quantize else "fp32_baseline")
    append_report_to_log(args.log_file, label, result)
    print(f"Saved as '{label}' to {args.log_file}")


if __name__ == "__main__":
    main()
