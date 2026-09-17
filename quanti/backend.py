from __future__ import annotations

import platform

import torch

ARM_MACHINES = {"aarch64", "arm64", "armv7l"}


def default_quantized_engine() -> str:
    supported = torch.backends.quantized.supported_engines
    preferred = "qnnpack" if platform.machine().lower() in ARM_MACHINES else "fbgemm"

    if preferred in supported:
        return preferred
    if supported:
        return supported[0]
    raise RuntimeError("No supported PyTorch quantized backend engine found on this machine")
