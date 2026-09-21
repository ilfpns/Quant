from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from torch import nn


def default_onnx_filename(model: nn.Module, date: datetime | None = None) -> str:
    """"<모델 클래스명>_<YYYYMMDD>.onnx" 형식의 기본 파일명을 만든다."""
    date_str = (date or datetime.now()).strftime("%Y%m%d")
    return f"{type(model).__name__}_{date_str}.onnx"


def onnx_file_size_kb(path: str) -> float:
    return Path(path).stat().st_size / 1024


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    path: str,
    opset_version: int = 17,
) -> None:
    """PyTorch 모델을 ONNX 파일로 내보낸다. 배치 차원(dim 0)은 가변으로 둔다."""
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,  # 우리 모델(Linear/ReLU)엔 충분히 안정적인 구식 TorchScript 기반 exporter 사용
    )
