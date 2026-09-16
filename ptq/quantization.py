from __future__ import annotations

import platform

import torch
from torch import nn

ARM_MACHINES = {"aarch64", "arm64", "armv7l"}
DEFAULT_DYNAMIC_QUANTIZE_LAYERS = {nn.Linear, nn.LSTM}


def default_quantized_engine() -> str:
    supported = torch.backends.quantized.supported_engines
    preferred = "qnnpack" if platform.machine().lower() in ARM_MACHINES else "fbgemm"

    if preferred in supported:
        return preferred
    if supported:
        return supported[0]
    raise RuntimeError("No supported PyTorch quantized backend engine found on this machine")


def dynamic_quantize(model: nn.Module, layers: set[type[nn.Module]] | None = None) -> nn.Module:
    torch.backends.quantized.engine = default_quantized_engine()
    model.eval()
    target_layers = layers if layers is not None else DEFAULT_DYNAMIC_QUANTIZE_LAYERS
    return torch.quantization.quantize_dynamic(model, target_layers, dtype=torch.qint8)
