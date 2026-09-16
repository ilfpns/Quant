from __future__ import annotations

import torch

from ptq.model import SimpleLSTM
from ptq.quantization import dynamic_quantize
from ptq.report import measure, print_comparison


def main() -> None:
    torch.manual_seed(0)
    model = SimpleLSTM()
    model.eval()

    dummy_input = torch.randn(1, 30, 128)

    before = measure("FP32 (before)", model)

    quantized_model = dynamic_quantize(model)
    after = measure("INT8 dynamic (after)", quantized_model)

    with torch.no_grad():
        fp32_output = model(dummy_input)
        quantized_output = quantized_model(dummy_input)
    max_diff = (fp32_output - quantized_output).abs().max().item()

    print_comparison(before, after, max_diff)


if __name__ == "__main__":
    main()
