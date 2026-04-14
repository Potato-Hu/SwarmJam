from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class TrainingLogger:
    """Persist compact training artifacts for debugging and quick inspection."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "history.csv"
        self.json_path = self.log_dir / "history.json"

    def write_history(self, history: list[dict[str, Any]]) -> None:
        if not history:
            return

        fieldnames = list(history[0].keys())
        with self.csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

        with self.json_path.open("w", encoding="utf-8") as json_file:
            json.dump(history, json_file, ensure_ascii=False, indent=2)

    def read_history(self) -> list[dict[str, Any]]:
        if not self.json_path.exists():
            return []
        with self.json_path.open("r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
        return payload if isinstance(payload, list) else []
