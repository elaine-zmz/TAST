from __future__ import annotations

import torch
import torch.nn as nn


class InstanceConditionedRefinement(nn.Module):
    """Task-specific residual correction in the second tower hidden space."""

    def __init__(
        self,
        input_dim: int,
        correction_dim: int,
        num_tasks: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        correction_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.correction_dim = int(correction_dim)
        self.num_tasks = int(num_tasks)
        self.hidden_dim = int(hidden_dim)
        self.correction_scale = float(correction_scale)
        if min(self.input_dim, self.correction_dim, self.num_tasks, self.hidden_dim) <= 0:
            raise ValueError("Refinement dimensions and num_tasks must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.correction_scale <= 0:
            raise ValueError("correction_scale must be positive")

        self.task_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.input_dim),
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.hidden_dim, self.correction_dim),
                )
                for _ in range(self.num_tasks)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module_group in self.task_modules:
            for module in module_group.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            output_layer = module_group[-1]
            if not isinstance(output_layer, nn.Linear):
                raise RuntimeError("The final refinement module must be Linear")
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        task_id: int,
        intermediate_representation: torch.Tensor,
    ) -> torch.Tensor:
        task_id = int(task_id)
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id={task_id} outside [0, {self.num_tasks})")
        if (
            intermediate_representation.ndim != 2
            or intermediate_representation.size(1) != self.input_dim
        ):
            raise ValueError(
                f"Expected representation [B, {self.input_dim}], "
                f"got {tuple(intermediate_representation.shape)}"
            )
        correction = self.task_modules[task_id](intermediate_representation)
        return self.correction_scale * torch.tanh(correction)

    def task_parameter_count(self, task_id: int) -> int:
        return int(
            sum(
                parameter.numel()
                for parameter in self.task_modules[int(task_id)].parameters()
            )
        )
