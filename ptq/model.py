from __future__ import annotations

import torch
from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 256, output_size: int = 10) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])
