from __future__ import annotations

import torch
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 256, output_size: int = 10) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.dequant(x)
