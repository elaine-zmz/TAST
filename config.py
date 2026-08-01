from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List

import torch


DATASET_TASKS = {
    "AliExpress_NL": ["ctr", "ctcvr"],
    "AliExpress_ES": ["ctr", "ctcvr"],
    "AliExpress_FR": ["ctr", "ctcvr"],
    "AliExpress_US": ["ctr", "ctcvr"],
    "UserBehavior": ["click", "buy", "cart", "favourite"],
}


@dataclass
class TrainingConfig:
    """Configuration for the two-stage TAST training procedure."""

    data_name: str = "AliExpress_NL"
    data_root: str = "./data"
    output_root: str = "./outputs_tast"
    seeds: List[int] | None = None
    device: str = "cuda:0"
    batch_size: int = 32768
    num_workers: int = 4
    val_ratio: float = 0.5
    split_seed: int = 42
    deterministic_algorithms: bool = True
    overwrite_existing: bool = False

    num_tasks: int | None = None
    task_names: List[str] | None = None
    embed_dim: int = 128
    backbone_layer_dims: List[int] | None = None
    tower_layer_dims: List[int] | None = None
    task_embed_dim: int = 16
    topology_condition_dim: int = 32
    topology_rank: int = 4
    topology_projector_hidden_dim: int = 64
    backbone_dropout: float = 0.2
    tower_dropout: float = 0.1

    connection_density_candidates: List[float] | None = None
    neuron_density_candidates: List[float] | None = None
    global_connection_density_budget: float = 0.60
    global_neuron_density_budget: float = 0.70
    density_allocator_hidden_dim: int = 32
    density_temperature_start: float = 1.5
    density_temperature_end: float = 0.5
    mask_temperature_start: float = 1.5
    mask_temperature_end: float = 0.7
    lambda_budget: float = 1e-1

    refinement_hidden_dim: int = 32
    refinement_dropout: float = 0.10
    correction_scale: float = 0.10
    tower_refinement_lr: float = 2e-5
    task_refinement_min_gain: float = 1e-4
    stage2_init_seed: int = 91021

    stage1_epochs: int = 40
    stage2_epochs: int = 5
    stage1_lr: float = 2e-4
    stage2_lr: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    stage2_patience: int = 2
    min_delta: float = 1e-5
    gradient_clip_norm: float = 1.0
    print_freq: int = 100

    def __post_init__(self) -> None:
        if self.data_name not in DATASET_TASKS:
            raise ValueError(
                f"Unsupported data_name={self.data_name}; "
                f"choose from {sorted(DATASET_TASKS)}"
            )

        if self.seeds is None:
            self.seeds = [2022, 2023, 2024, 2025, 2026]
        self.seeds = [int(seed) for seed in self.seeds]

        expected_tasks = list(DATASET_TASKS[self.data_name])
        self.task_names = (
            expected_tasks
            if self.task_names is None
            else [str(value) for value in self.task_names]
        )
        if self.task_names != expected_tasks:
            raise ValueError(
                f"{self.data_name} task_names must be {expected_tasks}, "
                f"got {self.task_names}"
            )

        self.num_tasks = (
            len(self.task_names) if self.num_tasks is None else int(self.num_tasks)
        )
        if self.num_tasks != len(self.task_names):
            raise ValueError("num_tasks must equal len(task_names)")

        self.backbone_layer_dims = (
            [256, 128]
            if self.backbone_layer_dims is None
            else [int(value) for value in self.backbone_layer_dims]
        )
        self.tower_layer_dims = (
            [128, 64]
            if self.tower_layer_dims is None
            else [int(value) for value in self.tower_layer_dims]
        )
        if not self.backbone_layer_dims:
            raise ValueError("backbone_layer_dims must not be empty")
        if len(self.tower_layer_dims) != 2:
            raise ValueError("tower_layer_dims must contain exactly two dimensions")

        self.connection_density_candidates = (
            [0.4, 0.5, 0.6, 0.7]
            if self.connection_density_candidates is None
            else [float(value) for value in self.connection_density_candidates]
        )
        self.neuron_density_candidates = (
            [0.5, 0.6, 0.7, 0.8]
            if self.neuron_density_candidates is None
            else [float(value) for value in self.neuron_density_candidates]
        )

        for name, values in (
            ("connection_density_candidates", self.connection_density_candidates),
            ("neuron_density_candidates", self.neuron_density_candidates),
        ):
            if not values or any(not 0.0 < value <= 1.0 for value in values):
                raise ValueError(f"{name} must contain values in (0, 1]")
            if sorted(set(values)) != values:
                raise ValueError(f"{name} must be sorted and unique")

        for name in (
            "global_connection_density_budget",
            "global_neuron_density_budget",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")

        if self.stage1_epochs <= 0 or self.stage2_epochs < 0:
            raise ValueError("stage1_epochs must be positive and stage2_epochs non-negative")
        if min(self.stage1_lr, self.stage2_lr, self.tower_refinement_lr) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.refinement_hidden_dim <= 0:
            raise ValueError("refinement_hidden_dim must be positive")
        if not 0.0 <= self.refinement_dropout < 1.0:
            raise ValueError("refinement_dropout must lie in [0, 1)")
        if self.correction_scale <= 0:
            raise ValueError("correction_scale must be positive")
        if self.task_refinement_min_gain < 0:
            raise ValueError("task_refinement_min_gain must be non-negative")
        if not 0.0 <= self.backbone_dropout < 1.0:
            raise ValueError("backbone_dropout must lie in [0, 1)")
        if not 0.0 <= self.tower_dropout < 1.0:
            raise ValueError("tower_dropout must lie in [0, 1)")
        if self.lambda_budget < 0:
            raise ValueError("lambda_budget must be non-negative")

    @property
    def total_epochs(self) -> int:
        return self.stage1_epochs + self.stage2_epochs

    def resolve_device(self) -> str:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, but CUDA is unavailable"
            )
        return self.device

    def to_dict(self) -> dict:
        result = asdict(self)
        result["total_epochs"] = self.total_epochs
        result["resolved_device"] = self.resolve_device()
        return result

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_args(cls) -> "TrainingConfig":
        parser = argparse.ArgumentParser(
            description="TAST task-specific topology and local-refinement trainer"
        )
        parser.add_argument("--data_name", default="AliExpress_NL", choices=sorted(DATASET_TASKS))
        parser.add_argument("--data_root", default="./data")
        parser.add_argument("--output_root", default="./outputs_tast")
        parser.add_argument("--seeds", type=int, nargs="+", default=[2022])
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--batch_size", type=int, default=32768)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--val_ratio", type=float, default=0.5)
        parser.add_argument("--split_seed", type=int, default=42)
        parser.add_argument("--overwrite_existing", action="store_true")
        parser.add_argument("--non_deterministic", action="store_true")

        parser.add_argument("--embed_dim", type=int, default=128)
        parser.add_argument("--backbone_layer_dims", type=int, nargs="+", default=[256, 128])
        parser.add_argument("--tower_layer_dims", type=int, nargs="+", default=[128, 64])
        parser.add_argument("--task_embed_dim", type=int, default=16)
        parser.add_argument("--topology_condition_dim", type=int, default=32)
        parser.add_argument("--topology_rank", type=int, default=4)
        parser.add_argument("--topology_projector_hidden_dim", type=int, default=64)
        parser.add_argument("--backbone_dropout", type=float, default=0.2)
        parser.add_argument("--tower_dropout", type=float, default=0.1)

        parser.add_argument(
            "--connection_density_candidates",
            type=float,
            nargs="+",
            default=[0.4, 0.5, 0.6, 0.7],
        )
        parser.add_argument(
            "--neuron_density_candidates",
            type=float,
            nargs="+",
            default=[0.5, 0.6, 0.7, 0.8],
        )
        parser.add_argument("--global_connection_density_budget", type=float, default=0.60)
        parser.add_argument("--global_neuron_density_budget", type=float, default=0.70)
        parser.add_argument("--density_allocator_hidden_dim", type=int, default=32)
        parser.add_argument("--density_temperature_start", type=float, default=1.5)
        parser.add_argument("--density_temperature_end", type=float, default=0.5)
        parser.add_argument("--mask_temperature_start", type=float, default=1.5)
        parser.add_argument("--mask_temperature_end", type=float, default=0.7)
        parser.add_argument("--lambda_budget", type=float, default=1e-1)

        parser.add_argument("--refinement_hidden_dim", type=int, default=32)
        parser.add_argument("--refinement_dropout", type=float, default=0.10)
        parser.add_argument("--correction_scale", type=float, default=0.10)
        parser.add_argument("--tower_refinement_lr", type=float, default=2e-5)
        parser.add_argument("--task_refinement_min_gain", type=float, default=1e-4)
        parser.add_argument("--stage2_init_seed", type=int, default=91021)

        parser.add_argument("--stage1_epochs", type=int, default=40)
        parser.add_argument("--stage2_epochs", type=int, default=5)
        parser.add_argument("--stage1_lr", type=float, default=2e-4)
        parser.add_argument("--stage2_lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument("--early_stopping_patience", type=int, default=10)
        parser.add_argument("--stage2_patience", type=int, default=2)
        parser.add_argument("--min_delta", type=float, default=1e-5)
        parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
        parser.add_argument("--print_freq", type=int, default=100)

        values = vars(parser.parse_args())
        values["deterministic_algorithms"] = not values.pop("non_deterministic")
        declared = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - declared)
        if unknown:
            raise TypeError(f"Arguments not declared in TrainingConfig: {unknown}")
        return cls(**values)
