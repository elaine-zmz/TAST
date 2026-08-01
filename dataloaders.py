from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset


USER_BEHAVIOR_FEATURE_COLUMNS = [
    "user_id:token",
    "item_id:token",
    "category:token",
]
USER_BEHAVIOR_LABEL_COLUMNS = [
    "click:label",
    "buy:label",
    "cart:label",
    "favourite:label",
]

def _to_tensor(value, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)

class EncodedTabularDataset(TorchDataset):
    def __init__(self, cached: Dict, expected_num_tasks: int, split: str) -> None:
        super().__init__()
        self.split = split
        self.categorical_data = _to_tensor(cached["categorical_data"], torch.long)
        self.numerical_data = _to_tensor(cached["numerical_data"], torch.float32)
        self.labels = _to_tensor(cached["labels"], torch.float32)
        self.numerical_num = int(cached["numerical_num"])
        self.field_dims = np.asarray(cached["field_dims"], dtype=np.int64)
        if self.labels.ndim != 2 or self.labels.size(1) != int(expected_num_tasks):
            raise ValueError(
                f"Expected {expected_num_tasks} labels, got {tuple(self.labels.shape)} in {split}."
            )

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, index: int):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]


class AliExpressDataset(EncodedTabularDataset):
    """Loads pre-encoded AliExpress CSV files or tensor caches."""

    def __init__(self, data_root: str | Path, dataset_name: str, split: str) -> None:
        data_dir = Path(data_root) / dataset_name
        cache_path = data_dir / f"{split}_cached.pt"
        csv_path = data_dir / f"{split}.csv"
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        cached = _load_cache(cache_path)
        if cached is None:
            if not csv_path.exists():
                raise FileNotFoundError(f"Neither cache nor CSV exists for split '{split}': {csv_path}")
            cached = self._build_cache(csv_path, cache_path)
        super().__init__(cached, expected_num_tasks=2, split=split)

    @staticmethod
    def _build_cache(csv_path: Path, cache_path: Path) -> Dict:
        frame = pd.read_csv(csv_path)
        if frame.shape[1] < 19:
            raise ValueError(f"Expected index + 16 categorical + two labels in {csv_path}.")
        values = frame.iloc[:, 1:].copy()
        categorical = values.iloc[:, :16].fillna(0).to_numpy(dtype=np.int64)
        numerical_frame = values.iloc[:, 16:-2].apply(pd.to_numeric, errors="coerce")
        if numerical_frame.shape[1] > 0:
            medians = numerical_frame.median(axis=0).fillna(0.0)
            numerical = numerical_frame.fillna(medians).to_numpy(dtype=np.float32)
        else:
            numerical = np.zeros((len(frame), 0), dtype=np.float32)
        labels = values.iloc[:, -2:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if np.any(categorical < 0):
            raise ValueError("Categorical IDs must be non-negative.")
        field_dims = (categorical.max(axis=0) + 1).astype(np.int64)
        cached = _make_cache(categorical, numerical, labels, field_dims)
        _atomic_torch_save(cached, cache_path)
        return cached


class UserBehaviorDataset(EncodedTabularDataset):
    """Four-task UserBehavior split with three token fields and no numerical fields."""

    def __init__(self, data_root: str | Path, split: str, field_dims: np.ndarray) -> None:
        data_dir = Path(data_root) / "UserBehavior"
        cache_path = data_dir / f"{split}_userbehavior_cached.pt"
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing UserBehavior split: {csv_path}")

        cached = _load_cache(cache_path)
        expected_dims = np.asarray(field_dims, dtype=np.int64)
        if cached is None or not np.array_equal(np.asarray(cached["field_dims"]), expected_dims):
            cached = self._build_cache(csv_path, cache_path, expected_dims)
        super().__init__(cached, expected_num_tasks=4, split=split)

    @staticmethod
    def _build_cache(csv_path: Path, cache_path: Path, field_dims: np.ndarray) -> Dict:
        frame = pd.read_csv(csv_path)
        missing = [
            column
            for column in (*USER_BEHAVIOR_FEATURE_COLUMNS, *USER_BEHAVIOR_LABEL_COLUMNS)
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        categorical_frame = frame[USER_BEHAVIOR_FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
        categorical = categorical_frame.to_numpy(dtype=np.int64)
        if np.any(categorical < 0):
            raise ValueError(f"Categorical IDs must be non-negative in {csv_path}.")
        for field_id, dimension in enumerate(field_dims):
            if categorical.shape[0] and int(categorical[:, field_id].max()) >= int(dimension):
                raise ValueError(
                    f"Field {USER_BEHAVIOR_FEATURE_COLUMNS[field_id]} exceeds shared field_dims in {csv_path}."
                )

        labels = frame[USER_BEHAVIOR_LABEL_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError(f"UserBehavior labels must be binary in {csv_path}.")
        numerical = np.zeros((len(frame), 0), dtype=np.float32)
        cached = _make_cache(categorical, numerical, labels, field_dims)
        _atomic_torch_save(cached, cache_path)
        return cached


def _load_cache(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        try:
            cached = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            cached = torch.load(path, map_location="cpu")
    except Exception as error:
        print(f"Warning: failed to load {path}: {error}; rebuilding from CSV.")
        return None
    required = {"categorical_data", "numerical_data", "labels", "numerical_num", "field_dims"}
    return cached if isinstance(cached, dict) and required.issubset(cached) else None


def _make_cache(categorical, numerical, labels, field_dims) -> Dict:
    return {
        "categorical_data": torch.from_numpy(np.asarray(categorical, dtype=np.int64)),
        "numerical_data": torch.from_numpy(np.asarray(numerical, dtype=np.float32)),
        "labels": torch.from_numpy(np.asarray(labels, dtype=np.float32)),
        "numerical_num": int(np.asarray(numerical).shape[1]),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
    }


def _atomic_torch_save(value: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _userbehavior_field_dims(data_dir: Path) -> np.ndarray:
    maxima = np.zeros(len(USER_BEHAVIOR_FEATURE_COLUMNS), dtype=np.int64)
    for split in ("train", "val", "test"):
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing UserBehavior split: {csv_path}")
        frame = pd.read_csv(csv_path, usecols=USER_BEHAVIOR_FEATURE_COLUMNS)
        values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.int64)
        if np.any(values < 0):
            raise ValueError(f"Categorical IDs must be non-negative in {csv_path}.")
        if len(values):
            maxima = np.maximum(maxima, values.max(axis=0))
    return maxima + 1


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: list[int]
    field_dims: np.ndarray
    numerical_num: int
    dataset_statistics: Dict
    split_metadata: Dict


def _label_statistics(dataset: TorchDataset, task_names: Sequence[str]) -> Dict:
    if isinstance(dataset, Subset):
        labels = dataset.dataset.labels[torch.as_tensor(dataset.indices, dtype=torch.long)]
    else:
        labels = dataset.labels
    if labels.size(1) != len(task_names):
        raise ValueError(f"Label/task mismatch: labels={labels.size(1)}, task_names={list(task_names)}")
    result = {"samples": int(labels.size(0))}
    for task_id, task_name in enumerate(task_names):
        positives = int((labels[:, task_id] > 0.5).sum().item())
        result[f"{task_name}_positives"] = positives
        result[f"{task_name}_positive_rate"] = float(positives / max(labels.size(0), 1))
    return result


def _build_loaders(train_dataset, val_dataset, test_dataset, batch_size, loader_seed, num_workers, pin_memory):
    generator = torch.Generator().manual_seed(loader_seed)
    common = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "persistent_workers": bool(num_workers > 0),
    }
    return (
        DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, generator=generator, **common),
        DataLoader(val_dataset, batch_size=int(batch_size), shuffle=False, **common),
        DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False, **common),
    )


def load_data(
    batch_size: int,
    data_name: str,
    data_root: str = "./data",
    val_ratio: float = 0.5,
    split_seed: int = 42,
    loader_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    split_output_dir: Optional[str] = None,
    task_names: Optional[Sequence[str]] = None,
) -> DataBundle:
    if task_names is None:
        task_names = ["click", "buy", "cart", "favourite"] if data_name == "UserBehavior" else ["ctr", "ctcvr"]
    task_names = list(task_names)

    if data_name == "UserBehavior":
        expected = ["click", "buy", "cart", "favourite"]
        if task_names != expected:
            raise ValueError(f"UserBehavior task_names must be {expected}, got {task_names}")
        data_dir = Path(data_root) / data_name
        field_dims = _userbehavior_field_dims(data_dir)
        train_dataset = UserBehaviorDataset(data_root, "train", field_dims)
        val_dataset = UserBehaviorDataset(data_root, "val", field_dims)
        test_dataset = UserBehaviorDataset(data_root, "test", field_dims)
        split_metadata = {
            "strategy": "explicit_train_val_test_files",
            "split_seed": None,
            "label_order": task_names,
            "feature_columns": USER_BEHAVIOR_FEATURE_COLUMNS,
        }
    else:
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must lie in (0, 1).")
        train_dataset = AliExpressDataset(data_root, data_name, "train")
        data_dir = Path(data_root) / data_name
        if (data_dir / "val.csv").exists() or (data_dir / "val_cached.pt").exists():
            val_dataset = AliExpressDataset(data_root, data_name, "val")
            test_dataset = AliExpressDataset(data_root, data_name, "test")
            split_metadata = {"strategy": "explicit_val_and_test_files", "split_seed": split_seed}
        else:
            full_test = AliExpressDataset(data_root, data_name, "test")
            generator_np = np.random.RandomState(split_seed)
            indices = np.arange(len(full_test))
            generator_np.shuffle(indices)
            n_val = max(1, min(len(indices) - 1, int(round(len(indices) * val_ratio))))
            val_indices = indices[:n_val].tolist()
            test_indices = indices[n_val:].tolist()
            val_dataset = Subset(full_test, val_indices)
            test_dataset = Subset(full_test, test_indices)
            split_metadata = {
                "strategy": "deterministic_random_split_of_official_test",
                "split_seed": split_seed,
                "val_ratio": val_ratio,
                "val_size": len(val_indices),
                "test_size": len(test_indices),
            }
            if split_output_dir is not None:
                output = Path(split_output_dir)
                output.mkdir(parents=True, exist_ok=True)
                with (output / "split_indices.json").open("w", encoding="utf-8") as file:
                    json.dump({"val_indices": val_indices, "test_indices": test_indices}, file)

        for subset in (val_dataset, test_dataset):
            base = subset.dataset if isinstance(subset, Subset) else subset
            maximum = base.categorical_data.max(dim=0).values.cpu().numpy()
            if np.any(maximum >= train_dataset.field_dims):
                raise ValueError(
                    "Validation/test contains categorical IDs outside train-derived field_dims. "
                    "Use a shared preprocessing vocabulary with an explicit OOV index."
                )
        field_dims = train_dataset.field_dims

    train_loader, val_loader, test_loader = _build_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, loader_seed, num_workers, pin_memory
    )
    statistics = {
        "train": _label_statistics(train_dataset, task_names),
        "validation": _label_statistics(val_dataset, task_names),
        "test": _label_statistics(test_dataset, task_names),
        "categorical_fields": int(len(field_dims)),
        "numerical_features": int(train_dataset.numerical_num),
        "field_dims": np.asarray(field_dims).tolist(),
        "task_names": task_names,
    }
    if data_name == "UserBehavior":
        labels = train_dataset.labels
        click = labels[:, 0] > 0.5
        for task_id, task_name in enumerate(task_names[1:], start=1):
            positives = labels[:, task_id] > 0.5
            count = int((positives & ~click).sum().item())
            total = int(positives.sum().item())
            statistics["train"][f"{task_name}_without_click"] = count
            statistics["train"][f"{task_name}_without_click_rate"] = float(count / max(total, 1))

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=[1] * len(task_names),
        field_dims=np.asarray(field_dims, dtype=np.int64),
        numerical_num=train_dataset.numerical_num,
        dataset_statistics=statistics,
        split_metadata=split_metadata,
    )
