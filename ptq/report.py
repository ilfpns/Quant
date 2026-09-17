from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class ModelStats:
    label: str
    state_dict_size_mb: float
    structure: str


@dataclass
class SpeedResult:
    label: str
    avg_latency_ms: float
    max_diff: float | None = None


def measure(label: str, model: nn.Module) -> ModelStats:
    tmp_path = Path(f"_tmp_{label}.pt")
    torch.save(model.state_dict(), tmp_path)
    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    tmp_path.unlink()
    return ModelStats(label=label, state_dict_size_mb=size_mb, structure=str(model))


def measure_latency_ms(
    model: nn.Module,
    sample_input: torch.Tensor,
    iterations: int = 200,
    warmup: int = 20,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(sample_input)

        start = time.perf_counter()
        for _ in range(iterations):
            model(sample_input)
        elapsed = time.perf_counter() - start

    return elapsed / iterations * 1000.0


def max_output_diff(reference: torch.Tensor, other: torch.Tensor) -> float:
    return (reference - other).abs().max().item()


def print_size_report(entries: list[ModelStats]) -> None:
    print("===== Model Size / Structure =====")
    for entry in entries:
        print()
        print(f"[{entry.label}]")
        print(entry.structure)
        print(f"State dict size: {entry.state_dict_size_mb:.3f} MB")
    print()
    print("===================================")


def print_speed_table(results: list[SpeedResult]) -> None:
    baseline_ms = results[0].avg_latency_ms

    print()
    print("===== Inference Speed Comparison =====")
    header = f"{'Label':<22}{'Latency(ms)':>14}{'Speedup':>10}{'MaxDiff':>12}"
    print(header)
    print("-" * len(header))
    for result in results:
        speedup = baseline_ms / result.avg_latency_ms
        diff_text = f"{result.max_diff:.6f}" if result.max_diff is not None else "-"
        print(f"{result.label:<22}{result.avg_latency_ms:>14.4f}{speedup:>9.2f}x{diff_text:>12}")
    print("=======================================")


def print_accuracy_report(results: list[SpeedResult]) -> None:
    print()
    print("===== Max Output Diff vs FP32 =====")
    for result in results:
        if result.max_diff is None:
            continue
        print(f"{result.label:<22}: {result.max_diff:.6f}")
    print("====================================")
