from __future__ import annotations

from pathlib import Path

from src.visualization.matplotlib_style import configure_matplotlib_for_ieee_pdf

configure_matplotlib_for_ieee_pdf()

import matplotlib.pyplot as plt


def plot_training_history(history: list[dict[str, float]], save_path: str | Path) -> Path | None:
    if not history:
        return None

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    updates = [entry["update"] for entry in history]
    avg_rewards = [entry["average_step_reward"] for entry in history]
    actor_losses = [entry["actor_loss"] for entry in history]
    critic_losses = [entry["critic_loss"] for entry in history]
    eval_returns = [entry.get("eval_avg_return") for entry in history]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(updates, avg_rewards, color="#1f77b4", linewidth=2.0, label="Avg step reward")
    if any(value is not None for value in eval_returns):
        filtered_x = [update for update, value in zip(updates, eval_returns, strict=False) if value is not None]
        filtered_y = [value for value in eval_returns if value is not None]
        axes[0].plot(filtered_x, filtered_y, color="#d62728", linewidth=1.8, linestyle="--", label="Eval avg return")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(updates, actor_losses, color="#2ca02c", linewidth=2.0)
    axes[1].set_ylabel("Actor loss")
    axes[1].grid(alpha=0.25)

    axes[2].plot(updates, critic_losses, color="#ff7f0e", linewidth=2.0)
    axes[2].set_ylabel("Critic loss")
    axes[2].set_xlabel("Update")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    plt.savefig(f"{path}.pdf", bbox_inches="tight") if path.suffix.lower() != ".pdf" else plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path if path.suffix.lower() == ".pdf" else Path(f"{path}.pdf")
