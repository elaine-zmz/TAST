from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_density(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1], got {value}")
    return value


def _logistic_probability(
    logits: torch.Tensor,
    temperature: float,
    stochastic: bool,
) -> torch.Tensor:
    if stochastic:
        eps = torch.finfo(logits.dtype).eps
        uniform = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        noise = torch.log(uniform) - torch.log1p(-uniform)
        logits = logits + noise
    return torch.sigmoid(logits / float(temperature))


def _topk_mask(
    scores: torch.Tensor,
    keep: int,
    eligible: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct an exact-cardinality binary mask."""
    eligible_bool = (
        torch.ones_like(scores, dtype=torch.bool)
        if eligible is None
        else eligible.detach() >= 0.5
    )
    available = int(eligible_bool.sum().item())
    keep = min(max(int(keep), 0), available)
    hard = torch.zeros_like(scores)
    if keep == 0:
        return hard
    ranked = scores.masked_fill(~eligible_bool, torch.finfo(scores.dtype).min)
    indices = torch.topk(ranked.reshape(-1), k=keep, sorted=False).indices
    hard.reshape(-1)[indices] = 1.0
    return hard


def _straight_through_topk(
    probability: torch.Tensor,
    keep: int,
    eligible: torch.Tensor | None = None,
) -> torch.Tensor:
    eligible_float = (
        torch.ones_like(probability)
        if eligible is None
        else eligible.to(probability.dtype)
    )
    soft = probability * eligible_float
    hard = _topk_mask(soft, keep=keep, eligible=eligible_float)
    return hard.detach() - soft.detach() + soft


def _straight_through_choice(
    logits: torch.Tensor,
    temperature: float,
    deterministic: bool,
) -> torch.Tensor:
    if deterministic:
        hard = torch.zeros_like(logits)
        hard[torch.argmax(logits)] = 1.0
        return hard
    return F.gumbel_softmax(logits, tau=float(temperature), hard=True, dim=-1)


class TaskConditionedTopologyGenerator(nn.Module):
    """Generate task-conditioned connection and neuron scores for every layer."""

    def __init__(
        self,
        layer_dims: Sequence[int],
        num_tasks: int,
        task_embed_dim: int = 16,
        condition_dim: int = 32,
        rank: int = 4,
        projector_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.layer_dims = [int(value) for value in layer_dims]
        self.num_tasks = int(num_tasks)
        self.condition_dim = int(condition_dim)
        self.rank = int(rank)
        if len(self.layer_dims) < 2 or self.num_tasks <= 0 or self.rank <= 0:
            raise ValueError("Invalid topology-generator dimensions")

        self.task_embedding = nn.Embedding(self.num_tasks, int(task_embed_dim))
        self.task_encoder = nn.Sequential(
            nn.Linear(int(task_embed_dim), int(projector_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(projector_hidden_dim), self.condition_dim),
        )
        self.left_factor_projections = nn.ModuleList()
        self.right_factor_projections = nn.ModuleList()
        self.neuron_score_projections = nn.ModuleList()
        for input_dim, output_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            self.left_factor_projections.append(
                nn.Linear(self.condition_dim, output_dim * self.rank, bias=False)
            )
            self.right_factor_projections.append(
                nn.Linear(self.condition_dim, input_dim * self.rank, bias=False)
            )
            self.neuron_score_projections.append(
                nn.Linear(self.condition_dim, output_dim, bias=True)
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        for module in self.task_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        projections = [
            *self.left_factor_projections,
            *self.right_factor_projections,
            *self.neuron_score_projections,
        ]
        for module in projections:
            nn.init.xavier_uniform_(module.weight, gain=0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def task_representation(self, task_id: int) -> torch.Tensor:
        task_id = int(task_id)
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id={task_id} outside [0, {self.num_tasks})")
        index = torch.tensor(task_id, device=self.task_embedding.weight.device)
        return self.task_encoder(self.task_embedding(index))

    def forward(
        self,
        task_id: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        task_representation = self.task_representation(task_id)
        connection_scores: list[torch.Tensor] = []
        neuron_scores: list[torch.Tensor] = []
        layer_pairs = zip(self.layer_dims[:-1], self.layer_dims[1:])
        for (input_dim, output_dim), left_projection, right_projection, neuron_projection in zip(
            layer_pairs,
            self.left_factor_projections,
            self.right_factor_projections,
            self.neuron_score_projections,
        ):
            left = left_projection(task_representation).view(output_dim, self.rank)
            right = right_projection(task_representation).view(input_dim, self.rank)
            connection_scores.append(
                (left @ right.transpose(0, 1)) / math.sqrt(float(self.rank))
            )
            neuron_scores.append(neuron_projection(task_representation))
        return connection_scores, neuron_scores, task_representation


class TaskLayerDensityAllocator(nn.Module):
    """Select task- and layer-specific connection and neuron densities."""

    def __init__(
        self,
        condition_dim: int,
        num_layers: int,
        connection_candidates: Sequence[float],
        neuron_candidates: Sequence[float],
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.connection_candidates = [float(value) for value in connection_candidates]
        self.neuron_candidates = [float(value) for value in neuron_candidates]
        for value in self.connection_candidates:
            _validate_density(value, "connection density candidate")
        for value in self.neuron_candidates:
            _validate_density(value, "neuron density candidate")

        self.register_buffer(
            "connection_candidate_values",
            torch.tensor(self.connection_candidates),
        )
        self.register_buffer(
            "neuron_candidate_values",
            torch.tensor(self.neuron_candidates),
        )
        self.layer_embedding = nn.Embedding(self.num_layers, int(condition_dim))
        self.encoder = nn.Sequential(
            nn.Linear(2 * int(condition_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.connection_head = nn.Linear(int(hidden_dim), len(self.connection_candidates))
        self.neuron_head = nn.Linear(int(hidden_dim), len(self.neuron_candidates))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.layer_embedding.weight, std=0.02)
        for module in [*self.encoder, self.connection_head, self.neuron_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.connection_head.bias[len(self.connection_candidates) // 2] = 0.5
            self.neuron_head.bias[len(self.neuron_candidates) // 2] = 0.5

    def forward(
        self,
        task_representation: torch.Tensor,
        layer_id: int,
        temperature: float,
        deterministic: bool,
    ) -> Dict[str, torch.Tensor]:
        layer_index = torch.tensor(int(layer_id), device=task_representation.device)
        hidden = self.encoder(
            torch.cat(
                [task_representation, self.layer_embedding(layer_index)],
                dim=0,
            )
        )
        connection_logits = self.connection_head(hidden)
        neuron_logits = self.neuron_head(hidden)
        connection_choice = _straight_through_choice(
            connection_logits, temperature, deterministic
        )
        neuron_choice = _straight_through_choice(
            neuron_logits, temperature, deterministic
        )
        connection_values = self.connection_candidate_values.to(task_representation)
        neuron_values = self.neuron_candidate_values.to(task_representation)
        return {
            "connection_density": torch.sum(connection_choice * connection_values),
            "neuron_density": torch.sum(neuron_choice * neuron_values),
            "connection_choice": connection_choice,
            "neuron_choice": neuron_choice,
        }


class HierarchicalTopologyMask(nn.Module):
    """Construct neuron masks first and connection masks within eligible edges."""

    def __init__(
        self,
        generator: TaskConditionedTopologyGenerator,
        density_allocator: TaskLayerDensityAllocator,
        layer_dims: Sequence[int],
        num_tasks: int,
        global_connection_budget: float,
        global_neuron_budget: float,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.density_allocator = density_allocator
        self.layer_dims = [int(value) for value in layer_dims]
        self.num_tasks = int(num_tasks)
        self.num_layers = len(self.layer_dims) - 1
        self.global_connection_budget = _validate_density(
            global_connection_budget, "global_connection_budget"
        )
        self.global_neuron_budget = _validate_density(
            global_neuron_budget, "global_neuron_budget"
        )
        self.mask_temperature = 1.0
        self.density_temperature = 1.0
        self._last_density_records: dict[
            int, list[Dict[str, torch.Tensor | float]]
        ] = {}

        self.register_buffer("_masks_frozen", torch.tensor(False), persistent=True)
        for task_id in range(self.num_tasks):
            for layer_id, (input_dim, output_dim) in enumerate(
                zip(self.layer_dims[:-1], self.layer_dims[1:])
            ):
                self.register_buffer(
                    self._connection_name(task_id, layer_id),
                    torch.ones(output_dim, input_dim),
                    persistent=True,
                )
                self.register_buffer(
                    self._neuron_name(task_id, layer_id),
                    torch.ones(output_dim),
                    persistent=True,
                )
                self.register_buffer(
                    self._connection_density_name(task_id, layer_id),
                    torch.tensor(1.0),
                    persistent=True,
                )
                self.register_buffer(
                    self._neuron_density_name(task_id, layer_id),
                    torch.tensor(1.0),
                    persistent=True,
                )

    @staticmethod
    def _connection_name(task_id: int, layer_id: int) -> str:
        return f"fixed_connection_task{task_id}_layer{layer_id}"

    @staticmethod
    def _neuron_name(task_id: int, layer_id: int) -> str:
        return f"fixed_neuron_task{task_id}_layer{layer_id}"

    @staticmethod
    def _connection_density_name(task_id: int, layer_id: int) -> str:
        return f"fixed_connection_density_task{task_id}_layer{layer_id}"

    @staticmethod
    def _neuron_density_name(task_id: int, layer_id: int) -> str:
        return f"fixed_neuron_density_task{task_id}_layer{layer_id}"

    @property
    def masks_frozen(self) -> bool:
        return bool(self._masks_frozen.item())

    def set_temperatures(
        self,
        mask_temperature: float,
        density_temperature: float,
    ) -> None:
        if mask_temperature <= 0 or density_temperature <= 0:
            raise ValueError("temperatures must be positive")
        self.mask_temperature = float(mask_temperature)
        self.density_temperature = float(density_temperature)

    def fixed_masks(
        self,
        task_id: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        connections = [
            getattr(self, self._connection_name(task_id, layer_id))
            for layer_id in range(self.num_layers)
        ]
        neurons = [
            getattr(self, self._neuron_name(task_id, layer_id))
            for layer_id in range(self.num_layers)
        ]
        return connections, neurons

    @staticmethod
    def _density_selected_mask(
        probability: torch.Tensor,
        candidate_densities: Sequence[float],
        choice: torch.Tensor,
        eligible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        available = (
            probability.numel()
            if eligible is None
            else int((eligible.detach() >= 0.5).sum().item())
        )
        candidate_masks: list[torch.Tensor] = []
        for density in candidate_densities:
            keep = max(1, int(round(float(density) * available))) if available else 0
            candidate_masks.append(
                _straight_through_topk(probability, keep, eligible)
            )
        stacked = torch.stack(candidate_masks, dim=0)
        view_shape = [choice.numel()] + [1] * probability.ndim
        return torch.sum(stacked * choice.view(*view_shape), dim=0)

    def forward(
        self,
        task_id: int,
        deterministic: bool = False,
        return_metadata: bool = False,
    ):
        task_id = int(task_id)
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id={task_id} outside [0, {self.num_tasks})")

        if self.masks_frozen:
            connections, neurons = self.fixed_masks(task_id)
            if not return_metadata:
                return connections, neurons
            metadata = []
            previous_neuron = connections[0].new_ones(self.layer_dims[0])
            for layer_id, (connection, neuron) in enumerate(zip(connections, neurons)):
                eligible = neuron[:, None] * previous_neuron[None, :]
                metadata.append(
                    {
                        "connection_density": getattr(
                            self,
                            self._connection_density_name(task_id, layer_id),
                        ),
                        "neuron_density": getattr(
                            self,
                            self._neuron_density_name(task_id, layer_id),
                        ),
                        "eligible_connections": float(
                            (eligible >= 0.5).sum().item()
                        ),
                        "total_connections": float(connection.numel()),
                        "total_neurons": float(neuron.numel()),
                    }
                )
                previous_neuron = neuron
            return connections, neurons, metadata

        connection_scores, neuron_scores, task_representation = self.generator(task_id)
        stochastic = self.training and not deterministic
        connection_masks: list[torch.Tensor] = []
        neuron_masks: list[torch.Tensor] = []
        metadata: list[Dict[str, torch.Tensor | float]] = []
        previous_neuron = connection_scores[0].new_ones(self.layer_dims[0])

        for layer_id, (connection_score, neuron_score) in enumerate(
            zip(connection_scores, neuron_scores)
        ):
            neuron_probability = _logistic_probability(
                neuron_score, self.mask_temperature, stochastic
            )
            connection_probability = _logistic_probability(
                connection_score, self.mask_temperature, stochastic
            )
            density = self.density_allocator(
                task_representation,
                layer_id,
                self.density_temperature,
                deterministic=(deterministic or not self.training),
            )
            neuron_mask = self._density_selected_mask(
                neuron_probability,
                self.density_allocator.neuron_candidates,
                density["neuron_choice"],
            )

            eligibility = neuron_mask[:, None] * previous_neuron[None, :]
            hard_eligibility = (eligibility.detach() >= 0.5).to(
                connection_probability.dtype
            )
            connection_mask = self._density_selected_mask(
                connection_probability,
                self.density_allocator.connection_candidates,
                density["connection_choice"],
                eligible=hard_eligibility,
            )
            connection_mask = connection_mask * eligibility

            connection_masks.append(connection_mask)
            neuron_masks.append(neuron_mask)
            metadata.append(
                {
                    "connection_density": density["connection_density"],
                    "neuron_density": density["neuron_density"],
                    "eligible_connections": float(hard_eligibility.sum().item()),
                    "total_connections": float(connection_probability.numel()),
                    "total_neurons": float(neuron_probability.numel()),
                }
            )
            previous_neuron = neuron_mask

        self._last_density_records[task_id] = metadata
        if return_metadata:
            return connection_masks, neuron_masks, metadata
        return connection_masks, neuron_masks

    @torch.no_grad()
    def harden(self) -> None:
        for task_id in range(self.num_tasks):
            connections, neurons, metadata = self.forward(
                task_id,
                deterministic=True,
                return_metadata=True,
            )
            for layer_id, (connection, neuron, layer_metadata) in enumerate(
                zip(connections, neurons, metadata)
            ):
                getattr(
                    self, self._connection_name(task_id, layer_id)
                ).copy_((connection >= 0.5).float())
                getattr(
                    self, self._neuron_name(task_id, layer_id)
                ).copy_((neuron >= 0.5).float())
                getattr(
                    self, self._connection_density_name(task_id, layer_id)
                ).copy_(
                    torch.as_tensor(
                        layer_metadata["connection_density"],
                        device=connection.device,
                    ).detach()
                )
                getattr(
                    self, self._neuron_density_name(task_id, layer_id)
                ).copy_(
                    torch.as_tensor(
                        layer_metadata["neuron_density"],
                        device=connection.device,
                    ).detach()
                )
        self._masks_frozen.fill_(True)
        for parameter in self.generator.parameters():
            parameter.requires_grad_(False)
        for parameter in self.density_allocator.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        self._masks_frozen.fill_(False)
        for parameter in self.generator.parameters():
            parameter.requires_grad_(True)
        for parameter in self.density_allocator.parameters():
            parameter.requires_grad_(True)

    def budget_loss(self) -> torch.Tensor:
        zero = next(self.parameters()).new_zeros(())
        if self.masks_frozen:
            return zero
        for task_id in range(self.num_tasks):
            if task_id not in self._last_density_records:
                self.forward(task_id, deterministic=False)

        connection_active = zero
        connection_total = zero
        neuron_active = zero
        neuron_total = zero
        for task_id in range(self.num_tasks):
            for metadata in self._last_density_records[task_id]:
                connection_density = torch.as_tensor(
                    metadata["connection_density"], device=zero.device
                )
                neuron_density = torch.as_tensor(
                    metadata["neuron_density"], device=zero.device
                )
                eligible_connections = zero.new_tensor(
                    float(metadata["eligible_connections"])
                )
                neurons = zero.new_tensor(float(metadata["total_neurons"]))
                connection_active = (
                    connection_active + connection_density * eligible_connections
                )
                connection_total = connection_total + eligible_connections
                neuron_active = neuron_active + neuron_density * neurons
                neuron_total = neuron_total + neurons

        mean_connection_density = connection_active / connection_total.clamp_min(1.0)
        mean_neuron_density = neuron_active / neuron_total.clamp_min(1.0)
        return (
            mean_connection_density - self.global_connection_budget
        ).pow(2) + (
            mean_neuron_density - self.global_neuron_budget
        ).pow(2)

    @torch.no_grad()
    def statistics(self) -> Dict[str, Dict]:
        result: dict[str, Dict] = {}
        for task_id in range(self.num_tasks):
            if self.masks_frozen:
                connections, neurons, metadata = self.forward(
                    task_id, return_metadata=True
                )
            else:
                connections, neurons, metadata = self.forward(
                    task_id, deterministic=True, return_metadata=True
                )
            task_result: dict[str, Dict] = {}
            previous_neuron = connections[0].new_ones(self.layer_dims[0])
            for layer_id, (connection, neuron, layer_metadata) in enumerate(
                zip(connections, neurons, metadata)
            ):
                connection_hard = connection >= 0.5
                neuron_hard = neuron >= 0.5
                eligibility = (
                    neuron_hard[:, None]
                    & (previous_neuron >= 0.5)[None, :]
                )
                active_connections = int(connection_hard.sum().item())
                eligible_connections = int(eligibility.sum().item())
                task_result[f"layer_{layer_id}"] = {
                    "selected_connection_density": float(
                        torch.as_tensor(
                            layer_metadata["connection_density"]
                        ).item()
                    ),
                    "selected_neuron_density": float(
                        torch.as_tensor(layer_metadata["neuron_density"]).item()
                    ),
                    "hard_connection_density_within_eligible": float(
                        active_connections / max(eligible_connections, 1)
                    ),
                    "hard_connection_density_full_matrix": float(
                        active_connections / max(connection_hard.numel(), 1)
                    ),
                    "hard_neuron_density": float(neuron_hard.float().mean().item()),
                    "active_connections": active_connections,
                    "eligible_connections": eligible_connections,
                    "total_connections": int(connection_hard.numel()),
                    "active_neurons": int(neuron_hard.sum().item()),
                    "total_neurons": int(neuron_hard.numel()),
                }
                previous_neuron = neuron_hard.float()
            result[f"task_{task_id}"] = task_result
        return result

    @torch.no_grad()
    def connection_overlap_statistics(self) -> List[Dict]:
        records: list[Dict] = []
        for first_task in range(self.num_tasks):
            first_connections, _ = (
                self.fixed_masks(first_task)
                if self.masks_frozen
                else self.forward(first_task, deterministic=True)
            )
            for second_task in range(first_task + 1, self.num_tasks):
                second_connections, _ = (
                    self.fixed_masks(second_task)
                    if self.masks_frozen
                    else self.forward(second_task, deterministic=True)
                )
                for layer_id, (first_mask, second_mask) in enumerate(
                    zip(first_connections, second_connections)
                ):
                    first = first_mask >= 0.5
                    second = second_mask >= 0.5
                    intersection = int((first & second).sum().item())
                    union = int((first | second).sum().item())
                    records.append(
                        {
                            "task_a": first_task,
                            "task_b": second_task,
                            "layer": layer_id,
                            "jaccard": float(intersection / max(union, 1)),
                            "hamming_ratio": float(
                                (first != second).float().mean().item()
                            ),
                            "shared_edge_ratio_full": float(
                                intersection / max(first.numel(), 1)
                            ),
                            "task_a_private_ratio": float(
                                (first & ~second).sum().item()
                                / max(first.sum().item(), 1)
                            ),
                            "task_b_private_ratio": float(
                                (second & ~first).sum().item()
                                / max(second.sum().item(), 1)
                            ),
                        }
                    )
        return records


class TaskSpecificSparseBackbone(nn.Module):
    """Shared parameter network executed through task-specific sparse topology."""

    def __init__(
        self,
        layer_dims: Sequence[int],
        num_tasks: int,
        topology_generator: TaskConditionedTopologyGenerator,
        density_allocator: TaskLayerDensityAllocator,
        global_connection_budget: float,
        global_neuron_budget: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer_dims = [int(value) for value in layer_dims]
        self.num_layers = len(self.layer_dims) - 1
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for input_dim, output_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            weight = nn.Parameter(torch.empty(output_dim, input_dim))
            bias = nn.Parameter(torch.zeros(output_dim))
            nn.init.xavier_uniform_(weight)
            self.weights.append(weight)
            self.biases.append(bias)

        self.topology = HierarchicalTopologyMask(
            generator=topology_generator,
            density_allocator=density_allocator,
            layer_dims=self.layer_dims,
            num_tasks=num_tasks,
            global_connection_budget=global_connection_budget,
            global_neuron_budget=global_neuron_budget,
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(output_dim) for output_dim in self.layer_dims[1:-1]]
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        connection_masks, neuron_masks = self.topology(task_id)
        hidden = x
        for layer_id, (weight, bias, connection_mask, neuron_mask) in enumerate(
            zip(self.weights, self.biases, connection_masks, neuron_masks)
        ):
            hidden = F.linear(
                hidden,
                weight * connection_mask.to(weight.dtype),
                bias,
            )
            if layer_id < self.num_layers - 1:
                hidden = self.layer_norms[layer_id](hidden)
                hidden = F.relu(hidden)
            hidden = hidden * neuron_mask.to(hidden.dtype).unsqueeze(0)
            if layer_id < self.num_layers - 1:
                hidden = self.dropout(hidden)
        return hidden
