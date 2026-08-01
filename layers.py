from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """Field-wise categorical embeddings with field offsets."""

    def __init__(self, field_dims: np.ndarray, embed_dim: int) -> None:
        super().__init__()
        dims = np.asarray(field_dims, dtype=np.int64)
        if dims.ndim != 1 or np.any(dims <= 0):
            raise ValueError("field_dims must be a one-dimensional positive array")
        self.field_dims = dims
        self.embed_dim = int(embed_dim)
        self.embedding = nn.Embedding(int(dims.sum()), self.embed_dim)
        offsets = np.array((0, *np.cumsum(dims)[:-1]), dtype=np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets), persistent=True)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != len(self.field_dims):
            raise ValueError(
                f"Expected categorical input [B, {len(self.field_dims)}], "
                f"got {tuple(x.shape)}"
            )
        return self.embedding(x.long() + self.offsets.unsqueeze(0))


class PredictionTower(nn.Module):
    """Two-hidden-layer task-specific prediction tower."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [int(value) for value in hidden_dims]
        if len(dims) != 2:
            raise ValueError("hidden_dims must contain exactly two dimensions")
        if min(int(input_dim), *dims, int(output_dim)) <= 0:
            raise ValueError("All tower dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.input_dim = int(input_dim)
        self.hidden_dims = dims
        self.output_dim = int(output_dim)
        self.first_linear = nn.Linear(self.input_dim, dims[0])
        self.second_linear = nn.Linear(dims[0], dims[1])
        self.output_linear = nn.Linear(dims[1], self.output_dim)
        self.first_dropout = nn.Dropout(float(dropout))
        self.second_dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.first_linear, self.second_linear, self.output_linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward_first_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.first_dropout(torch.relu(self.first_linear(x)))

    def apply_second_post(self, pre_activation: torch.Tensor) -> torch.Tensor:
        return self.second_dropout(torch.relu(pre_activation))

    def forward_second_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_second_post(self.second_linear(x))

    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.forward_first_block(x)
        second = self.forward_second_block(first)
        return self.forward_output(second)
