from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class ModelStats:
    label: str
    state_dict_size_mb: float
    structure: str


def measure(label: str, model: nn.Module) -> ModelStats:
    tmp_path = Path(f"_tmp_{label}.pt")
    torch.save(model.state_dict(), tmp_path)
    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    tmp_path.unlink()
    return ModelStats(label=label, state_dict_size_mb=size_mb, structure=str(model))


def print_comparison(before: ModelStats, after: ModelStats, max_output_diff: float) -> None:
    size_reduction_pct = (1 - after.state_dict_size_mb / before.state_dict_size_mb) * 100

    print("===== PTQ Dynamic Quantization Report =====")
    print()
    print(f"[Before] {before.label}")
    print(before.structure)
    print(f"State dict size: {before.state_dict_size_mb:.3f} MB")
    print()
    print(f"[After] {after.label}")
    print(after.structure)
    print(f"State dict size: {after.state_dict_size_mb:.3f} MB")
    print()
    print(f"Size reduction     : {size_reduction_pct:.1f} %")
    print(f"Max output diff     : {max_output_diff:.6f}")
    print("=============================================")
