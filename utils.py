from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.total = 0.0
        self.count = 0

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)

    def update(self, value: float, n: int = 1) -> None:
        self.value = float(value)
        self.total += float(value) * int(n)
        self.count += int(n)


def set_random_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
    torch.backends.cudnn.benchmark = False
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, default=str)
