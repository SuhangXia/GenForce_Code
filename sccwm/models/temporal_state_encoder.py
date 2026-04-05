from __future__ import annotations

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = int(hidden_dim)
        self.gates = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 2, kernel_size=kernel_size, padding=padding)
        self.candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates = self.gates(torch.cat([x, h], dim=1))
        z, r = gates.chunk(2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        q = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * q


class ConvGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        cells: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            cells.append(ConvGRUCell(in_dim, hidden_dim, kernel_size=kernel_size))
            in_dim = hidden_dim
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.dim() != 5:
            raise ValueError(f"Expected x shape (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, _, h, w = x.shape
        hidden = [x.new_zeros((b, self.hidden_dim, h, w)) for _ in range(self.num_layers)]
        outputs: list[torch.Tensor] = []
        for step in range(t):
            cur = x[:, step]
            for layer_idx, cell in enumerate(self.cells):
                hidden[layer_idx] = cell(cur, hidden[layer_idx])
                cur = hidden[layer_idx]
            outputs.append(cur)
        return torch.stack(outputs, dim=1), hidden
