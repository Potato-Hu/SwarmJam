from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


"""Checkpoint helpers for the debug MAPPO trainer.

This module supports both regular periodic checkpoints and resume-from-checkpoint.
Use cases:
1. Normal training periodically calls `save_checkpoint(...)`.
2. If training is interrupted, call `load_checkpoint(...)` on the saved file and continue.
3. If training is stopped with Ctrl+C, the training loop should save an extra interrupt
   checkpoint before exiting. After that, you can resume from that interrupt checkpoint.

Important note about Ctrl+C:
- Resume is possible only from the latest checkpoint that was successfully written.
- In this project, the training loop also writes an interrupt checkpoint on KeyboardInterrupt,
  so Ctrl+C can usually be resumed from the saved interrupt file.
- Work since the last completed checkpoint save is not guaranteed to survive if the process
  is killed before the interrupt checkpoint is written.
"""


def save_checkpoint(
    path: str | Path,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    update: int,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
    value_normalizer_state: dict[str, float] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": int(update),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "value_normalizer_state": value_normalizer_state,
        "config": config,
        "extra": extra or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location)


def restore_training_state(
    path: str | Path,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    payload = load_checkpoint(path, map_location=map_location)
    actor.load_state_dict(payload["actor_state_dict"])
    critic.load_state_dict(payload["critic_state_dict"])
    actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    return payload


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return None

    candidates = sorted(directory.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def find_best_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return None

    explicit_best = directory / "mappo_best.pt"
    if explicit_best.exists():
        return explicit_best

    best_path: Path | None = None
    best_score = float("-inf")
    for candidate in sorted(directory.glob("*.pt")):
        try:
            payload = load_checkpoint(candidate, map_location="cpu")
        except Exception:
            continue
        extra = payload.get("extra", {}) if isinstance(payload.get("extra", {}), dict) else {}
        latest_stats = extra.get("latest_stats", {}) if isinstance(extra.get("latest_stats", {}), dict) else {}
        score = latest_stats.get("eval_avg_return")
        if score is None:
            continue
        score_value = float(score)
        if score_value > best_score:
            best_score = score_value
            best_path = candidate

    return best_path if best_path is not None else find_latest_checkpoint(directory)
