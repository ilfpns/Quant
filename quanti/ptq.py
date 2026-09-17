from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn

from quanti.backend import default_quantized_engine

DEFAULT_DYNAMIC_QUANTIZE_LAYERS = {nn.Linear, nn.LSTM}


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
