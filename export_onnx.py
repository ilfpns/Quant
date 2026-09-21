from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import torch

from quanti.export import export_to_onnx
from quanti.model import SimpleMLP

INPUT_SIZE = 128
ONNX_PATH = "simple_mlp_fp32.onnx"


def main() -> None:
    torch.manual_seed(0)
    model = SimpleMLP(input_size=INPUT_SIZE)
    model.eval()
    sample_input = torch.randn(1, INPUT_SIZE)

    export_to_onnx(model, sample_input, ONNX_PATH)
    print(f"[OK] exported: {ONNX_PATH}")

    # 1) 구조 검증 — ONNX 그래프 자체가 스펙에 맞게 유효한지
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("[OK] onnx.checker.check_model passed")

    # 2) 출력 검증 — PyTorch 결과와 ONNX Runtime 실행 결과가 같은지
    with torch.no_grad():
        torch_output = model(sample_input).numpy()

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": sample_input.numpy()})[0]

    max_diff = np.abs(torch_output - onnx_output).max()
    print(f"[OK] max diff (PyTorch vs ONNX Runtime): {max_diff:.8f}")


if __name__ == "__main__":
    main()
