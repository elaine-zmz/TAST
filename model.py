from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .layers import EmbeddingLayer, PredictionTower
from .refinement import InstanceConditionedRefinement
from .topology import (
    TaskConditionedTopologyGenerator,
    TaskLayerDensityAllocator,
    TaskSpecificSparseBackbone,
)


class TAST(nn.Module):
    """Task-Aware Sparse Topology learning for multi-task recommendation.

    Stage 1 learns task-specific neuron and connection masks over one shared
    parameter network. Stage 2 freezes the hardened topology and learns a
    task-specific, instance-conditioned residual correction in the second
    hidden space of each prediction tower. Validation performance determines
    whether each task deploys the base or refined path.
    """

    def __init__(
        self,
        num_tasks: int,
        num_classes: Sequence[int] | int,
        backbone_layer_dims: Sequence[int],
        tower_layer_dims: Sequence[int],
        categorical_field_dims: Optional[np.ndarray] = None,
        numerical_num: int = 0,
        embed_dim: int = 128,
        input_dim: Optional[int] = None,
        task_embed_dim: int = 16,
        topology_condition_dim: int = 32,
        topology_rank: int = 4,
        topology_projector_hidden_dim: int = 64,
        connection_density_candidates: Sequence[float] = (0.4, 0.5, 0.6, 0.7),
        neuron_density_candidates: Sequence[float] = (0.5, 0.6, 0.7, 0.8),
        global_connection_density_budget: float = 0.6,
        global_neuron_density_budget: float = 0.7,
        density_allocator_hidden_dim: int = 32,
        backbone_dropout: float = 0.2,
        tower_dropout: float = 0.1,
        refinement_hidden_dim: int = 32,
        refinement_dropout: float = 0.1,
        correction_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.embed_dim = int(embed_dim)
        self.numerical_num = int(numerical_num)
        self.backbone_layer_dims = [int(value) for value in backbone_layer_dims]
        self.tower_layer_dims = [int(value) for value in tower_layer_dims]
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if not self.backbone_layer_dims:
            raise ValueError("backbone_layer_dims must not be empty")
        if len(self.tower_layer_dims) != 2:
            raise ValueError("tower_layer_dims must contain exactly two dimensions")

        self.use_embedding = categorical_field_dims is not None
        if self.use_embedding:
            self.categorical_field_dims = np.asarray(
                categorical_field_dims, dtype=np.int64
            )
            self.embedding = EmbeddingLayer(self.categorical_field_dims, self.embed_dim)
            self.input_dim = int(
                len(self.categorical_field_dims) * self.embed_dim
                + self.numerical_num
            )
        else:
            self.categorical_field_dims = None
            if input_dim is None:
                raise ValueError("input_dim is required without categorical fields")
            self.embedding = None
            self.input_dim = int(input_dim)

        backbone_dims = [self.input_dim, *self.backbone_layer_dims]
        self.representation_dim = int(backbone_dims[-1])
        topology_generator = TaskConditionedTopologyGenerator(
            layer_dims=backbone_dims,
            num_tasks=self.num_tasks,
            task_embed_dim=task_embed_dim,
            condition_dim=topology_condition_dim,
            rank=topology_rank,
            projector_hidden_dim=topology_projector_hidden_dim,
        )
        density_allocator = TaskLayerDensityAllocator(
            condition_dim=topology_condition_dim,
            num_layers=len(backbone_dims) - 1,
            connection_candidates=connection_density_candidates,
            neuron_candidates=neuron_density_candidates,
            hidden_dim=density_allocator_hidden_dim,
        )
        self.shared_backbone = TaskSpecificSparseBackbone(
            layer_dims=backbone_dims,
            num_tasks=self.num_tasks,
            topology_generator=topology_generator,
            density_allocator=density_allocator,
            global_connection_budget=global_connection_density_budget,
            global_neuron_budget=global_neuron_density_budget,
            dropout=backbone_dropout,
        )

        self.output_dims = (
            [int(num_classes)] * self.num_tasks
            if isinstance(num_classes, int)
            else [int(value) for value in num_classes]
        )
        if len(self.output_dims) != self.num_tasks:
            raise ValueError("num_classes must provide one output size per task")

        self.prediction_towers = nn.ModuleList(
            [
                PredictionTower(
                    input_dim=self.representation_dim,
                    hidden_dims=self.tower_layer_dims,
                    output_dim=self.output_dims[task_id],
                    dropout=tower_dropout,
                )
                for task_id in range(self.num_tasks)
            ]
        )
        self.base_towers = copy.deepcopy(self.prediction_towers)
        self._freeze_base_towers()

        self.first_tower_dim = int(self.tower_layer_dims[0])
        self.second_tower_dim = int(self.tower_layer_dims[1])
        self.local_refinement = InstanceConditionedRefinement(
            input_dim=self.first_tower_dim,
            correction_dim=self.second_tower_dim,
            num_tasks=self.num_tasks,
            hidden_dim=refinement_hidden_dim,
            dropout=refinement_dropout,
            correction_scale=correction_scale,
        )
        self.register_buffer(
            "refinement_enabled",
            torch.zeros(self.num_tasks, dtype=torch.bool),
            persistent=True,
        )
        self.model_mode = "stage1"

    @property
    def topology_generator(self) -> nn.Module:
        return self.shared_backbone.topology.generator

    @property
    def density_allocator(self) -> nn.Module:
        return self.shared_backbone.topology.density_allocator

    def _freeze_base_towers(self) -> None:
        self.base_towers.eval()
        for parameter in self.base_towers.parameters():
            parameter.requires_grad_(False)

    def snapshot_base_towers(self) -> None:
        self.base_towers.load_state_dict(self.prediction_towers.state_dict(), strict=True)
        self._freeze_base_towers()
        self.refinement_enabled.zero_()

    def restore_prediction_towers_from_base(self) -> None:
        self.prediction_towers.load_state_dict(self.base_towers.state_dict(), strict=True)

    def set_refinement_enabled(self, enabled: Sequence[bool]) -> None:
        values = torch.as_tensor(
            list(enabled),
            dtype=torch.bool,
            device=self.refinement_enabled.device,
        )
        if values.numel() != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} refinement flags, got {values.numel()}"
            )
        self.refinement_enabled.copy_(values)

    def model_spec(self) -> Dict:
        return {
            "num_tasks": self.num_tasks,
            "num_classes": list(self.output_dims),
            "backbone_layer_dims": list(self.backbone_layer_dims),
            "tower_layer_dims": list(self.tower_layer_dims),
            "categorical_field_dims": (
                None
                if self.categorical_field_dims is None
                else self.categorical_field_dims.tolist()
            ),
            "numerical_num": self.numerical_num,
            "embed_dim": self.embed_dim,
            "input_dim": None if self.use_embedding else self.input_dim,
            "refinement_hidden_dim": self.local_refinement.hidden_dim,
            "correction_scale": self.local_refinement.correction_scale,
        }

    def set_model_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode not in {"stage1", "refinement"}:
            raise ValueError("mode must be stage1 or refinement")
        self.model_mode = mode

    def set_stage1_temperatures(
        self,
        mask_temperature: float,
        density_temperature: float,
    ) -> None:
        self.shared_backbone.topology.set_temperatures(
            mask_temperature,
            density_temperature,
        )

    def harden_topology(self) -> None:
        self.shared_backbone.topology.harden()

    def reset_refinement_modules(self, seed: int) -> None:
        parameter = next(self.parameters())
        devices = [parameter.device] if parameter.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            self.restore_prediction_towers_from_base()
            self.local_refinement.reset_parameters()
            self.refinement_enabled.zero_()

    def encode_inputs(
        self,
        categorical_x: torch.Tensor,
        numerical_x: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_embedding:
            if self.embedding is None:
                raise RuntimeError("Embedding module is missing")
            embedded = self.embedding(categorical_x).flatten(start_dim=1)
            return torch.cat([embedded, numerical_x.float()], dim=1).float()
        return torch.cat(
            [categorical_x.float(), numerical_x.float()], dim=1
        ).float()

    @staticmethod
    def _squeeze_binary_output(output: torch.Tensor) -> torch.Tensor:
        return (
            output.squeeze(-1)
            if output.ndim == 2 and output.size(-1) == 1
            else output
        )

    def _tower_path(
        self,
        tower: PredictionTower,
        representation: torch.Tensor,
        hidden_correction: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_hidden = tower.forward_first_block(representation)
        second_pre_activation = tower.second_linear(first_hidden)
        if hidden_correction is not None:
            if hidden_correction.shape != second_pre_activation.shape:
                raise ValueError(
                    f"Correction shape {tuple(hidden_correction.shape)} does not "
                    f"match second hidden shape {tuple(second_pre_activation.shape)}"
                )
            second_pre_activation = second_pre_activation + hidden_correction
        second_hidden = tower.apply_second_post(second_pre_activation)
        logits = self._squeeze_binary_output(tower.forward_output(second_hidden))
        return first_hidden, second_hidden, logits

    def forward(
        self,
        categorical_x: torch.Tensor,
        numerical_x: torch.Tensor,
        task_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        inputs = self.encode_inputs(categorical_x, numerical_x)
        selected_tasks = (
            list(range(self.num_tasks))
            if task_ids is None
            else [int(value) for value in task_ids]
        )
        outputs: Dict[str, List[torch.Tensor]] = {
            key: []
            for key in (
                "refined_logits",
                "selected_logits",
                "base_logits",
                "hidden_corrections",
                "effective_logit_deltas",
                "task_representations",
                "intermediate_representations",
            )
        }

        for task_id in selected_tasks:
            task_representation = self.shared_backbone(inputs, task_id)
            if self.model_mode == "stage1":
                intermediate, _, base_logits = self._tower_path(
                    self.prediction_towers[task_id],
                    task_representation,
                )
                refined_logits = base_logits
                hidden_correction = intermediate.new_zeros(
                    intermediate.size(0), self.second_tower_dim
                )
            else:
                frozen_representation = task_representation.detach()
                intermediate, _, base_logits = self._tower_path(
                    self.base_towers[task_id],
                    frozen_representation,
                )
                hidden_correction = self.local_refinement(
                    task_id,
                    intermediate.detach(),
                )
                _, _, refined_logits = self._tower_path(
                    self.prediction_towers[task_id],
                    frozen_representation,
                    hidden_correction,
                )

            selected_logits = (
                refined_logits
                if bool(self.refinement_enabled[task_id].item())
                else base_logits
            )
            outputs["refined_logits"].append(refined_logits)
            outputs["selected_logits"].append(selected_logits)
            outputs["base_logits"].append(base_logits)
            outputs["hidden_corrections"].append(hidden_correction)
            outputs["effective_logit_deltas"].append(
                refined_logits - base_logits.detach()
            )
            outputs["task_representations"].append(task_representation)
            outputs["intermediate_representations"].append(intermediate)
        return outputs

    def predict_selected(
        self,
        categorical_x: torch.Tensor,
        numerical_x: torch.Tensor,
        task_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Execute only the validation-selected path for each task."""
        inputs = self.encode_inputs(categorical_x, numerical_x)
        selected_tasks = (
            list(range(self.num_tasks))
            if task_ids is None
            else [int(value) for value in task_ids]
        )
        selected_logits: list[torch.Tensor] = []
        for task_id in selected_tasks:
            task_representation = self.shared_backbone(inputs, task_id)
            if self.model_mode == "stage1":
                _, _, logits = self._tower_path(
                    self.prediction_towers[task_id], task_representation
                )
            elif bool(self.refinement_enabled[task_id].item()):
                frozen_representation = task_representation.detach()
                intermediate = self.prediction_towers[
                    task_id
                ].forward_first_block(frozen_representation)
                correction = self.local_refinement(
                    task_id, intermediate.detach()
                )
                second_pre_activation = self.prediction_towers[
                    task_id
                ].second_linear(intermediate) + correction
                second_hidden = self.prediction_towers[
                    task_id
                ].apply_second_post(second_pre_activation)
                logits = self._squeeze_binary_output(
                    self.prediction_towers[task_id].forward_output(second_hidden)
                )
            else:
                _, _, logits = self._tower_path(
                    self.base_towers[task_id], task_representation.detach()
                )
            selected_logits.append(logits)
        return {"selected_logits": selected_logits}

    def topology_budget_loss(self) -> torch.Tensor:
        return self.shared_backbone.topology.budget_loss()

    @torch.no_grad()
    def topology_statistics(self) -> Dict:
        return self.shared_backbone.topology.statistics()

    @torch.no_grad()
    def connection_overlap_statistics(self) -> List[Dict]:
        return self.shared_backbone.topology.connection_overlap_statistics()

    @staticmethod
    def _linear_operations(module: nn.Module) -> int:
        return int(
            sum(
                layer.in_features * layer.out_features
                for layer in module.modules()
                if isinstance(layer, nn.Linear)
            )
        )

    @torch.no_grad()
    def parameter_statistics(self) -> Dict:
        topology = self.shared_backbone.topology
        per_task = []
        for task_id in range(self.num_tasks):
            connections, neurons = (
                topology.fixed_masks(task_id)
                if topology.masks_frozen
                else topology(task_id, deterministic=True)
            )
            active_connections = sum(
                int((connection >= 0.5).sum().item())
                for connection in connections
            )
            active_neurons = sum(
                int((neuron >= 0.5).sum().item()) for neuron in neurons
            )
            per_task.append(
                {
                    "task": task_id,
                    "refinement_enabled": bool(
                        self.refinement_enabled[task_id].item()
                    ),
                    "active_connections": active_connections,
                    "active_neurons": active_neurons,
                    "tower_parameters": int(
                        sum(
                            parameter.numel()
                            for parameter in self.prediction_towers[
                                task_id
                            ].parameters()
                        )
                    ),
                    "refinement_parameters": self.local_refinement.task_parameter_count(
                        task_id
                    ),
                    "dense_backbone_operations": int(
                        sum(weight.numel() for weight in self.shared_backbone.weights)
                    ),
                    "tower_operations": self._linear_operations(
                        self.prediction_towers[task_id]
                    ),
                }
            )

        return {
            "total_parameters": int(
                sum(parameter.numel() for parameter in self.parameters())
            ),
            "trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                )
            ),
            "embedding_parameters": int(
                0
                if self.embedding is None
                else sum(parameter.numel() for parameter in self.embedding.parameters())
            ),
            "shared_backbone_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.shared_backbone.parameters()
                )
            ),
            "topology_generator_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.topology_generator.parameters()
                )
            ),
            "density_allocator_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.density_allocator.parameters()
                )
            ),
            "prediction_tower_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.prediction_towers.parameters()
                )
            ),
            "base_tower_reference_parameters": int(
                sum(parameter.numel() for parameter in self.base_towers.parameters())
            ),
            "local_refinement_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.local_refinement.parameters()
                )
            ),
            "model_mode": self.model_mode,
            "refinement_enabled": self.refinement_enabled.cpu().tolist(),
            "per_task": per_task,
        }
