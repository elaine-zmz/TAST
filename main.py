from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .data import load_data
from .model import TAST
from .trainer import TASTTrainer
from .utils import save_json, set_random_seed


def build_model(
    config: TrainingConfig,
    field_dims,
    numerical_num: int,
    num_classes,
) -> TAST:
    return TAST(
        num_tasks=config.num_tasks,
        num_classes=num_classes,
        backbone_layer_dims=config.backbone_layer_dims,
        tower_layer_dims=config.tower_layer_dims,
        categorical_field_dims=field_dims,
        numerical_num=numerical_num,
        embed_dim=config.embed_dim,
        task_embed_dim=config.task_embed_dim,
        topology_condition_dim=config.topology_condition_dim,
        topology_rank=config.topology_rank,
        topology_projector_hidden_dim=config.topology_projector_hidden_dim,
        connection_density_candidates=config.connection_density_candidates,
        neuron_density_candidates=config.neuron_density_candidates,
        global_connection_density_budget=config.global_connection_density_budget,
        global_neuron_density_budget=config.global_neuron_density_budget,
        density_allocator_hidden_dim=config.density_allocator_hidden_dim,
        backbone_dropout=config.backbone_dropout,
        tower_dropout=config.tower_dropout,
        refinement_hidden_dim=config.refinement_hidden_dim,
        refinement_dropout=config.refinement_dropout,
        correction_scale=config.correction_scale,
    )


def aggregate_results(results: list[dict], output_dir: Path) -> None:
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "seed_level_results.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number])
    summary = {}
    for column in numeric.columns:
        if column in {"seed", "best_epoch", "accepted_stage"}:
            continue
        summary[column] = {
            "mean": float(numeric[column].mean()),
            "std": (
                float(numeric[column].std(ddof=1)) if len(numeric) > 1 else 0.0
            ),
            "runs": int(len(numeric)),
        }
    save_json(summary, output_dir / "mean_std_summary.json")
    print(json.dumps(summary, indent=2))


def main() -> None:
    config = TrainingConfig.from_args()
    experiment_dir = Path(config.output_root) / config.data_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config.save(experiment_dir / "experiment_config.json")

    results: list[dict] = []
    for seed in config.seeds:
        print(
            f"\n{'=' * 80}\n"
            f"Dataset={config.data_name} Seed={seed}\n"
            f"{'=' * 80}"
        )
        run_dir = experiment_dir / f"seed_{seed}"
        result_path = run_dir / "result.json"
        if result_path.exists() and not config.overwrite_existing:
            print(f"Skipping completed run: {result_path}")
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        set_random_seed(seed, config.deterministic_algorithms)
        run_dir.mkdir(parents=True, exist_ok=True)
        config.save(run_dir / "config.json")
        bundle = load_data(
            batch_size=config.batch_size,
            data_name=config.data_name,
            data_root=config.data_root,
            val_ratio=config.val_ratio,
            split_seed=config.split_seed,
            loader_seed=seed,
            num_workers=config.num_workers,
            pin_memory=config.resolve_device().startswith("cuda"),
            split_output_dir=str(run_dir),
            task_names=config.task_names,
        )
        save_json(bundle.dataset_statistics, run_dir / "dataset_statistics.json")
        save_json(bundle.split_metadata, run_dir / "split_metadata.json")

        model = build_model(
            config,
            bundle.field_dims,
            bundle.numerical_num,
            bundle.num_classes,
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"Backbone={config.backbone_layer_dims} "
            f"Tower={config.tower_layer_dims} "
            f"Tasks={config.task_names} "
            f"Parameters={total_parameters:,}"
        )
        trainer = TASTTrainer(
            model=model,
            train_loader=bundle.train_loader,
            val_loader=bundle.val_loader,
            test_loader=bundle.test_loader,
            config=config,
            run_dir=run_dir,
            seed=seed,
        )
        results.append(trainer.train())

    aggregate_results(results, experiment_dir)


if __name__ == "__main__":
    main()
