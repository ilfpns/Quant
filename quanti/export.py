from __future__ import annotations

import torch
from torch import nn


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
