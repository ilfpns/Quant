# Quant

## Person Detection (YOLOv8 + Webcam)

Real-time person detection from a local webcam using YOLOv8.

### Setup

```bash
pip install -r requirements.txt
```

The first run downloads `yolov8n.pt` automatically.

### Run

```bash
python main.py
```

Options:

- `--camera-index` (default `0`)
- `--model` (default `yolov8n.pt`)
- `--confidence` (default `0.5`)
- `--width` / `--height` (default `1280x720`)

Press `q` to quit, `c` to print a benchmark report (FPS, avg inference latency, model size, avg CPU usage) for everything measured so far.

### Structure

```
person_detector/
  camera.py      # webcam capture (Camera)
  detector.py     # YOLOv8 inference, filtered to the "person" class (PersonDetector, Detection)
  visualizer.py   # draws bounding boxes / labels on frames (draw_detections)
  metrics.py      # FPS / latency / model size / CPU tracking (PerformanceMonitor, print_report)
  app.py          # wires camera + detector + visualizer + metrics into the run loop (PersonDetectionApp)
main.py           # CLI entry point for the live viewer
computer_info.py  # CLI entry point for a headless benchmark run
```

### Benchmark

```bash
python computer_info.py
```

Runs the camera + detector for a fixed duration (default 15s, `--duration`) and reports FPS, average per-frame inference latency (ms), model file size (MB), and average CPU usage (`psutil`, skip with `--no-cpu`).

### Component Guidelines

Keep responsibilities isolated so each piece can be swapped or tested independently:

- `Camera` only knows how to open/read/close a video source. It has no knowledge of detection or rendering.
- `PersonDetector` only knows how to turn a frame into a list of `Detection`s. It has no knowledge of the camera or display.
- `draw_detections` only knows how to render `Detection`s onto a frame. It has no side effects and returns a new frame.
- `PersonDetectionApp` is the only place that wires these together into a loop.

When extending this project (e.g. multi-camera support, a different model, tracking, saving video), add or modify a single component rather than mixing concerns into `app.py` or `main.py`.
