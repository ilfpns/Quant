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


def build_self_distillation_data(
    reference_model: nn.Module,
    input_size: int,
    batches: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """FP32 모델의 출력을 타깃으로 삼아 QAT 학습용 (input, target) 쌍을 생성한다.

    실제 라벨링된 데이터셋이 없으므로, FP32 모델을 teacher로 사용하는
    self-distillation 방식으로 QAT가 FP32 출력을 최대한 재현하도록 학습시킨다.
    """
    reference_model.eval()
    data: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for _ in range(batches):
            x = torch.randn(1, input_size)
            y = reference_model(x)
            data.append((x, y))
    return data


def qat_quantize(
    model: nn.Module,
    train_data: Iterable[tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 3,
    lr: float = 1e-3,
) -> nn.Module:
    """학습 중 fake-quant를 삽입해 양자화 오차를 모델 스스로 보정하도록 학습(QAT)한 뒤 변환한다."""
    engine = default_quantized_engine()
    torch.backends.quantized.engine = engine

    model = copy.deepcopy(model)
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig(engine)

    prepared = torch.quantization.prepare_qat(model, inplace=False)

    optimizer = torch.optim.Adam(prepared.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for inputs, targets in train_data:
            optimizer.zero_grad()
            outputs = prepared(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    prepared.eval()
    return torch.quantization.convert(prepared, inplace=False)
