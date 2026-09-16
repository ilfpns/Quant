from __future__ import annotations

import copy
import platform
from typing import Iterable

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
    model = copy.deepcopy(model)
    model.eval()
    target_layers = layers if layers is not None else DEFAULT_DYNAMIC_QUANTIZE_LAYERS
    return torch.quantization.quantize_dynamic(model, target_layers, dtype=torch.qint8)


def static_quantize(model: nn.Module, calibration_data: Iterable[torch.Tensor]) -> nn.Module:
    engine = default_quantized_engine()
    torch.backends.quantized.engine = engine

    model = copy.deepcopy(model)
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(engine)

    prepared = torch.quantization.prepare(model, inplace=False)
    with torch.no_grad():
        for sample in calibration_data:
            prepared(sample)

    return torch.quantization.convert(prepared, inplace=False)
