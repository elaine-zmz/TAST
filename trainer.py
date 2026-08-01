from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import TrainingConfig
from .metrics import binary_metrics
from .model import TAST
from .utils import AverageMeter, save_json


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class TASTTrainer:
    """Train task-specific sparse topology and local refinement in two stages."""

    def __init__(
        self,
        model: TAST,
        train_loader,
        val_loader,
        test_loader,
        config: TrainingConfig,
        run_dir: str | Path,
        seed: int,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.seed = int(seed)
        self.device = torch.device(config.resolve_device())
        self.model.to(self.device)

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage1_checkpoint = self.run_dir / "best_stage1.pt"
        self.stage2_candidate_checkpoint = self.run_dir / "best_stage2_candidate.pt"
        self.stage2_checkpoint = self.run_dir / "best_stage2.pt"
        self.best_checkpoint = self.run_dir / "best_model.pt"

        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: list[dict] = []
        self.stage_parameter_report: Dict[str, Dict[str, int]] = {}
        self.stage1_best_metrics: Dict[str, float] = {}
        self.stage2_best_metrics: Dict[str, float] = {}
        self.stage2_candidate_metrics: Dict[str, float] = {}
        self.stage2_candidate_epoch = -1
        self.stage2_candidate_score = float("-inf")
        self.best_val_metrics: Dict[str, float] = {}
        self.best_epoch = -1
        self.accepted_stage = 1
        self.training_seconds = 0.0

    @staticmethod
    def _set_requires_grad(module: nn.Module | None, flag: bool) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad_(flag)

    @staticmethod
    def _unique_trainable(
        parameters: Iterable[torch.nn.Parameter],
    ) -> List[torch.nn.Parameter]:
        result: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for parameter in parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                result.append(parameter)
                seen.add(id(parameter))
        return result

    def _freeze_all(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _module_counts(self) -> Dict[str, int]:
        modules = {
            "embedding": self.model.embedding,
            "shared_backbone": self.model.shared_backbone,
            "topology_generator": self.model.topology_generator,
            "density_allocator": self.model.density_allocator,
            "prediction_towers": self.model.prediction_towers,
            "base_towers": self.model.base_towers,
            "local_refinement": self.model.local_refinement,
        }
        return {
            name: (
                0
                if module is None
                else int(
                    sum(
                        parameter.numel()
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    )
                )
            )
            for name, module in modules.items()
        }

    def _configure_stage1(self) -> None:
        self.model.set_model_mode("stage1")
        topology = self.model.shared_backbone.topology
        if topology.masks_frozen:
            topology.unfreeze()

        self._freeze_all()
        self._set_requires_grad(self.model.embedding, True)
        self._set_requires_grad(self.model.shared_backbone, True)
        self._set_requires_grad(self.model.prediction_towers, True)
        self._set_requires_grad(self.model.base_towers, False)
        self._set_requires_grad(self.model.local_refinement, False)

        parameters = self._unique_trainable(self.model.parameters())
        if not parameters:
            raise RuntimeError("Stage 1 has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.stage1_lr,
            weight_decay=self.config.weight_decay,
        )
        self.stage_parameter_report["stage1"] = self._module_counts()
        print("Configured Stage 1 optimizer", self.stage_parameter_report["stage1"])

    def _configure_stage2(self) -> None:
        self.model.set_model_mode("refinement")
        self._freeze_all()

        self._set_requires_grad(self.model.local_refinement, True)
        refinement_parameters = self._unique_trainable(
            self.model.local_refinement.parameters()
        )

        tower_parameters: list[torch.nn.Parameter] = []
        for tower in self.model.prediction_towers:
            self._set_requires_grad(tower.second_linear, True)
            self._set_requires_grad(tower.output_linear, True)
            tower_parameters.extend(
                self._unique_trainable(tower.second_linear.parameters())
            )
            tower_parameters.extend(
                self._unique_trainable(tower.output_linear.parameters())
            )

        parameter_groups = []
        if refinement_parameters:
            parameter_groups.append(
                {"params": refinement_parameters, "lr": self.config.stage2_lr}
            )
        if tower_parameters:
            parameter_groups.append(
                {
                    "params": tower_parameters,
                    "lr": self.config.tower_refinement_lr,
                }
            )
        if not parameter_groups:
            raise RuntimeError("Stage 2 has no trainable parameters")

        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.config.weight_decay,
        )
        self.stage_parameter_report["stage2"] = self._module_counts()
        print("Configured Stage 2 optimizer", self.stage_parameter_report["stage2"])

    @staticmethod
    def _linear_schedule(
        start: float,
        end: float,
        epoch: int,
        epochs: int,
    ) -> float:
        progress = min(1.0, float(epoch) / max(int(epochs) - 1, 1))
        return float(start) + progress * (float(end) - float(start))

    def _task_loss(
        self,
        logits: Sequence[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if len(logits) != self.model.num_tasks:
            raise ValueError(
                f"Expected {self.model.num_tasks} task logits, got {len(logits)}"
            )
        losses = [
            self.loss_function(logit, targets[:, task_id].float())
            for task_id, logit in enumerate(logits)
        ]
        return torch.stack(losses).sum()

    def _stage1_loss(
        self,
        outputs: Dict,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        task_loss = self._task_loss(outputs["refined_logits"], targets)
        budget_loss = self.model.topology_budget_loss()
        total = task_loss + self.config.lambda_budget * budget_loss
        return total, {
            "task_loss": task_loss.detach(),
            "budget_loss": budget_loss.detach(),
        }

    def _stage2_loss(
        self,
        outputs: Dict,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        task_losses = [
            self.loss_function(
                outputs["refined_logits"][task_id],
                targets[:, task_id].float(),
            )
            for task_id in range(self.model.num_tasks)
        ]
        total = torch.stack(task_losses).sum()
        hidden_correction_mse = torch.stack(
            [
                correction.square().mean()
                for correction in outputs["hidden_corrections"]
            ]
        ).mean()
        logit_delta_mse = torch.stack(
            [
                delta.square().mean()
                for delta in outputs["effective_logit_deltas"]
            ]
        ).mean()
        parts: Dict[str, torch.Tensor] = {
            "task_loss": total.detach(),
            "hidden_correction_mse": hidden_correction_mse.detach(),
            "logit_delta_mse": logit_delta_mse.detach(),
        }
        for task_id, task_loss in enumerate(task_losses):
            parts[f"{self.config.task_names[task_id]}_bce"] = task_loss.detach()
        return total, parts

    def _loss(
        self,
        outputs: Dict,
        targets: torch.Tensor,
        stage: str,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if stage == "stage1":
            return self._stage1_loss(outputs, targets)
        if stage == "stage2":
            return self._stage2_loss(outputs, targets)
        raise ValueError(f"Unknown stage={stage}")

    def _set_train_modes(self, stage: str) -> None:
        self.model.train()
        self.model.base_towers.eval()
        if stage == "stage2":
            if self.model.embedding is not None:
                self.model.embedding.eval()
            self.model.shared_backbone.eval()
            self.model.prediction_towers.eval()
            self.model.local_refinement.train()

    def train_epoch(
        self,
        epoch: int,
        stage: str,
        local_epoch: int,
        total_local_epochs: int,
    ) -> Dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not configured")
        self._set_train_modes(stage)

        if stage == "stage1":
            mask_temperature = self._linear_schedule(
                self.config.mask_temperature_start,
                self.config.mask_temperature_end,
                local_epoch,
                total_local_epochs,
            )
            density_temperature = self._linear_schedule(
                self.config.density_temperature_start,
                self.config.density_temperature_end,
                local_epoch,
                total_local_epochs,
            )
            self.model.set_stage1_temperatures(
                mask_temperature,
                density_temperature,
            )
            status_text = (
                f"mask_temp={mask_temperature:.3f} "
                f"density_temp={density_temperature:.3f}"
            )
        else:
            status_text = "local_refinement"

        meters: Dict[str, AverageMeter] = {"loss": AverageMeter()}
        start = time.time()
        for step, (categorical, numerical, targets) in enumerate(self.train_loader):
            categorical = categorical.to(self.device, non_blocking=True)
            numerical = numerical.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = int(categorical.size(0))

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(categorical, numerical)
            loss, parts = self._loss(outputs, targets, stage)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at stage={stage}, epoch={epoch}, step={step}"
                )
            loss.backward()
            if self.config.gradient_clip_norm > 0:
                trainable_gradients = [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(
                    trainable_gradients,
                    self.config.gradient_clip_norm,
                )
            self.optimizer.step()

            meters["loss"].update(float(loss.item()), batch_size)
            for name, value in parts.items():
                meters.setdefault(name, AverageMeter()).update(
                    float(value.detach().cpu()), batch_size
                )
            if step % self.config.print_freq == 0:
                details = " ".join(
                    f"{name}={float(value):.5f}"
                    for name, value in parts.items()
                )
                print(
                    f"Epoch {epoch:03d} {stage} [{step}/{len(self.train_loader)}] "
                    f"loss={loss.item():.5f} avg={meters['loss'].average:.5f} "
                    f"{status_text} {details}"
                )

        elapsed = time.time() - start
        self.training_seconds += elapsed
        print(f"Epoch {epoch:03d} {stage} time: {elapsed:.1f}s")
        return {name: meter.average for name, meter in meters.items()}

    @staticmethod
    def _append_probabilities(destination: list, logits: torch.Tensor) -> None:
        destination.extend(
            torch.sigmoid(logits).detach().cpu().numpy().tolist()
        )

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        split: str,
        use_saved_selection: bool = False,
    ) -> Dict[str, float]:
        self.model.eval()
        labels = [[] for _ in range(self.model.num_tasks)]
        predictions = {
            path: [[] for _ in range(self.model.num_tasks)]
            for path in ("base", "refined")
        }
        logit_delta_square_sum = [0.0] * self.model.num_tasks
        logit_delta_count = [0] * self.model.num_tasks
        hidden_correction_norm_sum = [0.0] * self.model.num_tasks
        sample_count = [0] * self.model.num_tasks

        for categorical, numerical, targets in loader:
            categorical = categorical.to(self.device, non_blocking=True)
            numerical = numerical.to(self.device, non_blocking=True)
            outputs = self.model(categorical, numerical)
            for task_id in range(self.model.num_tasks):
                target_cpu = targets[:, task_id].float().cpu()
                labels[task_id].extend(target_cpu.numpy().tolist())
                self._append_probabilities(
                    predictions["base"][task_id],
                    outputs["base_logits"][task_id],
                )
                self._append_probabilities(
                    predictions["refined"][task_id],
                    outputs["refined_logits"][task_id],
                )

                logit_delta = outputs["effective_logit_deltas"][task_id].detach().cpu()
                correction = outputs["hidden_corrections"][task_id].detach().cpu()
                logit_delta_square_sum[task_id] += float(
                    logit_delta.square().sum().item()
                )
                logit_delta_count[task_id] += int(logit_delta.numel())
                hidden_correction_norm_sum[task_id] += float(
                    correction.norm(dim=1).sum().item()
                )
                sample_count[task_id] += int(correction.size(0))

        result: Dict[str, float] = {}
        path_auc = {path: [] for path in predictions}
        for task_id, task_name in enumerate(self.config.task_names):
            for path in predictions:
                metrics = binary_metrics(
                    labels[task_id],
                    predictions[path][task_id],
                    f"{task_name}_{path}",
                )
                result.update(metrics)
                path_auc[path].append(float(metrics[f"{task_name}_{path}_auc"]))

        if use_saved_selection:
            enabled = self.model.refinement_enabled.cpu().tolist()
        else:
            enabled = [
                result[f"{task_name}_refined_auc"]
                >= result[f"{task_name}_base_auc"]
                + self.config.task_refinement_min_gain
                for task_name in self.config.task_names
            ]

        selected_auc = []
        for task_id, task_name in enumerate(self.config.task_names):
            selected_path = "refined" if enabled[task_id] else "base"
            selected_metrics = binary_metrics(
                labels[task_id],
                predictions[selected_path][task_id],
                task_name,
            )
            result.update(selected_metrics)
            result[f"{task_name}_refinement_enabled"] = int(enabled[task_id])
            result[f"{task_name}_selected_path"] = selected_path
            result[f"{task_name}_refined_gain_over_base"] = float(
                result[f"{task_name}_refined_auc"]
                - result[f"{task_name}_base_auc"]
            )
            delta_count = max(logit_delta_count[task_id], 1)
            task_samples = max(sample_count[task_id], 1)
            result[f"{task_name}_logit_delta_rms"] = float(
                np.sqrt(logit_delta_square_sum[task_id] / delta_count)
            )
            result[f"{task_name}_hidden_correction_norm_mean"] = float(
                hidden_correction_norm_sum[task_id] / task_samples
            )
            selected_auc.append(float(selected_metrics[f"{task_name}_auc"]))

        result["base_mean_auc"] = float(np.mean(path_auc["base"]))
        result["refined_mean_auc"] = float(np.mean(path_auc["refined"]))
        result["mean_auc"] = float(np.mean(selected_auc))
        result["selected_mean_auc"] = result["mean_auc"]
        result["selected_gain_over_base"] = float(
            result["selected_mean_auc"] - result["base_mean_auc"]
        )
        result["refined_gain_over_base"] = float(
            result["refined_mean_auc"] - result["base_mean_auc"]
        )
        result["enabled_task_count"] = int(sum(enabled))

        metric_text = " ".join(
            (
                f"{name}[base={result[f'{name}_base_auc']:.6f},"
                f"refined={result[f'{name}_refined_auc']:.6f},"
                f"selected={result[f'{name}_auc']:.6f},"
                f"enabled={result[f'{name}_refinement_enabled']}]"
            )
            for name in self.config.task_names
        )
        print(
            f"{split}: {metric_text} "
            f"selected_mean={result['selected_mean_auc']:.6f} "
            f"base={result['base_mean_auc']:.6f} "
            f"gain={result['selected_gain_over_base']:+.6f}"
        )
        return result

    def _checkpoint_payload(
        self,
        epoch: int,
        metrics: Dict[str, float],
        stage: int,
    ) -> Dict:
        topology = self.model.shared_backbone.topology
        return {
            "model": self.model.state_dict(),
            "optimizer": (
                None if self.optimizer is None else self.optimizer.state_dict()
            ),
            "epoch": int(epoch),
            "stage": int(stage),
            "seed": self.seed,
            "best_validation_metrics": metrics,
            "config": self.config.to_dict(),
            "model_spec": self.model.model_spec(),
            "model_mode": self.model.model_mode,
            "mask_temperature": float(topology.mask_temperature),
            "density_temperature": float(topology.density_temperature),
            "refinement_enabled": self.model.refinement_enabled.cpu().tolist(),
            "code_family": "tast-core",
        }

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float],
        stage: int,
    ) -> None:
        torch.save(self._checkpoint_payload(epoch, metrics, stage), path)

    def _load_checkpoint(self, path: Path) -> Dict:
        checkpoint = _torch_load(path, self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        default_mode = "stage1" if int(checkpoint.get("stage", 1)) == 1 else "refinement"
        self.model.set_model_mode(checkpoint.get("model_mode", default_mode))
        self.model.set_stage1_temperatures(
            float(
                checkpoint.get(
                    "mask_temperature", self.config.mask_temperature_end
                )
            ),
            float(
                checkpoint.get(
                    "density_temperature", self.config.density_temperature_end
                )
            ),
        )
        if "refinement_enabled" in checkpoint:
            self.model.set_refinement_enabled(checkpoint["refinement_enabled"])
        return checkpoint

    def _acceptable(
        self,
        candidate: Dict[str, float],
        baseline: Dict[str, float],
    ) -> bool:
        return (
            candidate.get("enabled_task_count", 0) > 0
            and candidate["selected_mean_auc"]
            > baseline["base_mean_auc"] + self.config.min_delta
        )

    def _record(
        self,
        epoch: int,
        stage: str,
        train_stats: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        row = {
            "epoch": int(epoch),
            "stage": stage,
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        self.history.append(row)
        keys = sorted({key for item in self.history for key in item})
        with (self.run_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    def _train_stage1(self) -> None:
        self._configure_stage1()
        best_score = float("-inf")
        no_improvement = 0
        for epoch in range(self.config.stage1_epochs):
            train_stats = self.train_epoch(
                epoch,
                "stage1",
                epoch,
                self.config.stage1_epochs,
            )
            validation = self.evaluate(self.val_loader, "Validation-Stage1")
            self._record(epoch, "stage1", train_stats, validation)
            if validation["mean_auc"] > best_score + self.config.min_delta:
                best_score = float(validation["mean_auc"])
                self.stage1_best_metrics = dict(validation)
                self.best_epoch = epoch
                self._save_checkpoint(
                    self.stage1_checkpoint,
                    epoch,
                    validation,
                    stage=1,
                )
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= self.config.early_stopping_patience:
                print("Early stopping in Stage 1")
                break

        if not self.stage1_checkpoint.exists():
            raise RuntimeError("Stage 1 did not save a checkpoint")

        checkpoint = self._load_checkpoint(self.stage1_checkpoint)
        self.model.harden_topology()
        self.model.snapshot_base_towers()
        self.model.set_model_mode("stage1")
        hardened_validation = self.evaluate(
            self.val_loader,
            "Validation-Stage1-Hardened",
        )
        self.stage1_best_metrics = dict(hardened_validation)
        self.best_epoch = int(checkpoint.get("epoch", self.best_epoch))
        self._save_checkpoint(
            self.stage1_checkpoint,
            self.best_epoch,
            hardened_validation,
            stage=1,
        )
        self._save_checkpoint(
            self.best_checkpoint,
            self.best_epoch,
            hardened_validation,
            stage=1,
        )
        self.best_val_metrics = dict(hardened_validation)
        self.accepted_stage = 1

    def _prepare_stage2_baseline(self) -> None:
        self.model.set_model_mode("stage1")
        self.model.snapshot_base_towers()
        baseline = self.evaluate(
            self.val_loader,
            "Validation-Stage1-Baseline",
        )
        self.stage1_best_metrics = dict(baseline)
        self._save_checkpoint(
            self.best_checkpoint,
            self.best_epoch,
            baseline,
            stage=1,
        )
        self.best_val_metrics = dict(baseline)
        self.accepted_stage = 1

    def _train_stage2(self, global_epoch: int) -> None:
        self._prepare_stage2_baseline()
        self.model.reset_refinement_modules(
            self.config.stage2_init_seed + self.seed
        )
        self.model.set_model_mode("refinement")
        self._configure_stage2()

        self.stage2_candidate_score = float("-inf")
        self.stage2_candidate_epoch = -1
        self.stage2_candidate_metrics = {}
        for path in (self.stage2_candidate_checkpoint, self.stage2_checkpoint):
            if path.exists():
                path.unlink()

        no_improvement = 0
        for local_epoch in range(self.config.stage2_epochs):
            epoch = global_epoch + local_epoch
            train_stats = self.train_epoch(
                epoch,
                "stage2",
                local_epoch,
                self.config.stage2_epochs,
            )
            validation = self.evaluate(self.val_loader, "Validation-Stage2")
            self._record(
                epoch,
                "stage2_local_refinement",
                train_stats,
                validation,
            )
            enabled = [
                bool(validation[f"{name}_refinement_enabled"])
                for name in self.config.task_names
            ]
            score = float(validation["selected_mean_auc"])
            if score > self.stage2_candidate_score + self.config.min_delta:
                self.model.set_refinement_enabled(enabled)
                self.stage2_candidate_score = score
                self.stage2_candidate_epoch = epoch
                self.stage2_candidate_metrics = dict(validation)
                self._save_checkpoint(
                    self.stage2_candidate_checkpoint,
                    epoch,
                    validation,
                    stage=2,
                )
                save_json(
                    validation,
                    self.run_dir / "stage2_candidate_metrics.json",
                )
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= self.config.stage2_patience:
                print("Early stopping in Stage 2")
                break

        if not self.stage2_candidate_checkpoint.exists():
            print("No Stage-2 candidate; retaining Stage 1")
            self._load_checkpoint(self.best_checkpoint)
            self.stage2_best_metrics = dict(self.stage1_best_metrics)
            return

        candidate = self._load_checkpoint(self.stage2_candidate_checkpoint)
        selected_validation = self.evaluate(
            self.val_loader,
            "Validation-Stage2-Best",
            use_saved_selection=True,
        )
        self.stage2_candidate_metrics = dict(selected_validation)
        save_json(
            selected_validation,
            self.run_dir / "stage2_candidate_metrics.json",
        )

        if self._acceptable(selected_validation, self.stage1_best_metrics):
            self.stage2_best_metrics = dict(selected_validation)
            self.best_val_metrics = dict(selected_validation)
            self.best_epoch = int(
                candidate.get("epoch", self.stage2_candidate_epoch)
            )
            self.accepted_stage = 2
            self._save_checkpoint(
                self.stage2_checkpoint,
                self.best_epoch,
                selected_validation,
                stage=2,
            )
            self._save_checkpoint(
                self.best_checkpoint,
                self.best_epoch,
                selected_validation,
                stage=2,
            )
            print(
                "Accepted Stage 2: "
                f"epoch={self.best_epoch} "
                f"selected_mean={selected_validation['selected_mean_auc']:.6f} "
                f"gain={selected_validation['selected_gain_over_base']:+.6f} "
                f"enabled={self.model.refinement_enabled.cpu().tolist()}"
            )
        else:
            print("Stage 2 did not improve validation performance; retaining Stage 1")
            self._load_checkpoint(self.best_checkpoint)
            self.stage2_best_metrics = dict(self.stage1_best_metrics)
            self.accepted_stage = 1

    @staticmethod
    def _extract_base_metrics(
        metrics: Dict[str, float],
        task_names: Sequence[str],
    ) -> Dict[str, float]:
        result: Dict[str, float] = {}
        auc_values = []
        for task_name in task_names:
            for suffix in (
                "auc",
                "logloss",
                "positive_rate",
                "prediction_mean",
                "prediction_std",
            ):
                source = f"{task_name}_base_{suffix}"
                if source in metrics:
                    result[f"{task_name}_{suffix}"] = float(metrics[source])
            auc_values.append(float(metrics[f"{task_name}_base_auc"]))
        result["mean_auc"] = float(np.mean(auc_values))
        return result

    def train(self) -> Dict[str, float]:
        self._train_stage1()
        if self.config.stage2_epochs > 0:
            self._load_checkpoint(self.stage1_checkpoint)
            self._train_stage2(global_epoch=self.config.stage1_epochs)

        save_json(
            self.stage_parameter_report,
            self.run_dir / "stage_trainable_parameters.json",
        )
        accepted = self._load_checkpoint(self.best_checkpoint)
        self.accepted_stage = int(accepted.get("stage", self.accepted_stage))
        self.best_epoch = int(accepted.get("epoch", self.best_epoch))

        if self.accepted_stage == 2:
            self.model.set_model_mode("refinement")
            test_metrics = self.evaluate(
                self.test_loader,
                "Test-Stage2",
                use_saved_selection=True,
            )
        else:
            self.model.set_model_mode("stage1")
            test_metrics = self.evaluate(self.test_loader, "Test-Stage1")

        test_stage1 = self._extract_base_metrics(
            test_metrics,
            self.config.task_names,
        )
        save_json(
            self.model.topology_statistics(),
            self.run_dir / "topology_statistics.json",
        )
        save_json(
            self.model.connection_overlap_statistics(),
            self.run_dir / "connection_overlap_statistics.json",
        )
        parameter_statistics = self.model.parameter_statistics()
        parameter_statistics["checkpoint_size_bytes"] = int(
            self.best_checkpoint.stat().st_size
        )
        save_json(
            parameter_statistics,
            self.run_dir / "parameter_statistics.json",
        )

        stage2_validation = (
            self.stage2_best_metrics.get(
                "selected_mean_auc",
                self.stage2_best_metrics.get("mean_auc"),
            )
            if self.stage2_best_metrics
            else None
        )
        result = {
            "seed": self.seed,
            "model_name": "tast",
            "best_epoch": self.best_epoch,
            "accepted_stage": self.accepted_stage,
            "backbone_layer_dims": list(self.config.backbone_layer_dims),
            "tower_layer_dims": list(self.config.tower_layer_dims),
            "refinement_hidden_dim": self.config.refinement_hidden_dim,
            "correction_scale": self.config.correction_scale,
            "tower_refinement_lr": self.config.tower_refinement_lr,
            "task_refinement_min_gain": self.config.task_refinement_min_gain,
            "refinement_enabled": self.model.refinement_enabled.cpu().tolist(),
            "training_seconds": float(self.training_seconds),
            "total_parameters": int(parameter_statistics["total_parameters"]),
            "checkpoint_size_bytes": int(
                parameter_statistics["checkpoint_size_bytes"]
            ),
            "stage1_val_mean_auc": (
                float(self.stage1_best_metrics.get("mean_auc"))
                if self.stage1_best_metrics
                else None
            ),
            "stage2_val_mean_auc": (
                None if stage2_validation is None else float(stage2_validation)
            ),
            "stage2_gain_over_stage1": (
                None
                if stage2_validation is None or not self.stage1_best_metrics
                else float(
                    stage2_validation - self.stage1_best_metrics["mean_auc"]
                )
            ),
            "stage2_accepted": bool(self.accepted_stage == 2),
            "stage2_candidate_best_epoch": self.stage2_candidate_epoch,
            "stage2_candidate_best_mean_auc": (
                float(self.stage2_candidate_score)
                if self.stage2_candidate_score > float("-inf")
                else None
            ),
            "test_stage1_mean_auc": float(test_stage1["mean_auc"]),
            "test_final_mean_auc": float(test_metrics["mean_auc"]),
            "test_final_gain_over_stage1": float(
                test_metrics["mean_auc"] - test_stage1["mean_auc"]
            ),
            **{
                f"stage2_candidate_{key}": value
                for key, value in self.stage2_candidate_metrics.items()
            },
            **{f"val_{key}": value for key, value in self.best_val_metrics.items()},
            **{
                f"test_stage1_{key}": value
                for key, value in test_stage1.items()
            },
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        save_json(result, self.run_dir / "result.json")
        return result
