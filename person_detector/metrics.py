from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class BenchmarkResult:
    frame_count: int
    elapsed_seconds: float
    latencies_ms: list[float]
    cpu_percentages: list[float]
    model_size_mb: float

    @property
    def fps(self) -> float:
        return self.frame_count / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def avg_cpu_percent(self) -> float | None:
        if not self.cpu_percentages:
            return None
        return sum(self.cpu_percentages) / len(self.cpu_percentages)


class PerformanceMonitor:
    def __init__(self, model_path: str, track_cpu: bool = True) -> None:
        self._model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        self._track_cpu = track_cpu and psutil is not None
        self._start_time: float | None = None
        self._latencies_ms: list[float] = []
        self._cpu_percentages: list[float] = []
        if self._track_cpu:
            psutil.cpu_percent(interval=None)

    @property
    def cpu_tracking_enabled(self) -> bool:
        return self._track_cpu

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def record_frame(self, inference_seconds: float) -> None:
        self._latencies_ms.append(inference_seconds * 1000.0)
        if self._track_cpu:
            self._cpu_percentages.append(psutil.cpu_percent(interval=None))

    def result(self) -> BenchmarkResult:
        elapsed = time.perf_counter() - self._start_time if self._start_time is not None else 0.0
        return BenchmarkResult(
            frame_count=len(self._latencies_ms),
            elapsed_seconds=elapsed,
            latencies_ms=self._latencies_ms,
            cpu_percentages=self._cpu_percentages,
            model_size_mb=self._model_size_mb,
        )


def print_report(result: BenchmarkResult, cpu_tracking_enabled: bool) -> None:
    print()
    print("===== Benchmark Report =====")
    print(f"Frames measured     : {result.frame_count}")
    print(f"Duration            : {result.elapsed_seconds:.2f} s")
    print(f"FPS                 : {result.fps:.2f}")
    print(f"Avg inference time  : {result.avg_latency_ms:.2f} ms")
    print(f"Model size          : {result.model_size_mb:.2f} MB")
    if result.avg_cpu_percent is not None:
        print(f"Avg CPU usage       : {result.avg_cpu_percent:.1f} %")
    elif cpu_tracking_enabled:
        print("Avg CPU usage       : (psutil not installed, skipped)")
    else:
        print("Avg CPU usage       : (disabled via --no-cpu)")
    print("=============================")
