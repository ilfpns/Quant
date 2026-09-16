from __future__ import annotations

import torch

from ptq.model import SimpleMLP
from ptq.quantization import dynamic_quantize, static_quantize
from ptq.report import (
    SpeedResult,
    max_output_diff,
    measure,
    measure_latency_ms,
    print_accuracy_report,
    print_size_report,
    print_speed_table,
)

INPUT_SIZE = 128
CALIBRATION_BATCHES = 32


def main() -> None:
    torch.manual_seed(0)
    fp32_model = SimpleMLP(input_size=INPUT_SIZE)
    fp32_model.eval()

    sample_input = torch.randn(1, INPUT_SIZE)
    calibration_data = [torch.randn(1, INPUT_SIZE) for _ in range(CALIBRATION_BATCHES)]

    dynamic_model = dynamic_quantize(fp32_model)
    static_model = static_quantize(fp32_model, calibration_data)

    with torch.no_grad():
        fp32_output = fp32_model(sample_input)
        dynamic_output = dynamic_model(sample_input)
        static_output = static_model(sample_input)

    print_size_report(
        [
            measure("FP32", fp32_model),
            measure("Dynamic PTQ (INT8)", dynamic_model),
            measure("Static PTQ (INT8)", static_model),
        ]
    )

    speed_results = [
        SpeedResult("FP32", measure_latency_ms(fp32_model, sample_input)),
        SpeedResult(
            "Dynamic PTQ",
            measure_latency_ms(dynamic_model, sample_input),
            max_output_diff(fp32_output, dynamic_output),
        ),
        SpeedResult(
            "Static PTQ",
            measure_latency_ms(static_model, sample_input),
            max_output_diff(fp32_output, static_output),
        ),
    ]

    print_speed_table(speed_results)
    print_accuracy_report(speed_results)


if __name__ == "__main__":
    main()
