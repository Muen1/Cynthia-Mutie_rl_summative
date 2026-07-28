"""
Convergence curves: trains ONE baseline run per algorithm while
periodically evaluating, so you get reward-over-training-progress (what
the rubric calls "convergence plots"), plus DQN's training loss curve
("DQN objective curves").

This is separate from the *_sweep.csv files (which only store each run's
FINAL score) — this script captures what happens DURING training.

Usage:
    uv run python -m training.convergence
Outputs (CSVs + a combined PNG) go to logs/ and assets/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from training.utils import make_env, log_run, LOGS_DIR, timer
from training.pg_training import PolicyNet

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
EVAL_FREQ = 2000
N_EVAL_EPISODES = 10


class LossLoggerCallback(BaseCallback):
    """Records DQN's training loss at every rollout so we can plot the
    'objective curve' the rubric asks for (SB3 doesn't expose this over
    time by default outside of tensorboard)."""

    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = csv_path

    def _on_step(self) -> bool:
        if self.n_calls % 500 == 0:
            loss = self.model.logger.name_to_value.get("train/loss")
            if loss is not None:
                log_run(self.csv_path, {
                    "timesteps": self.num_timesteps,
                    "loss": round(float(loss), 5),
                })
        return True


def train_with_eval_curve(algo_cls, algo_name: str, cfg: dict, timesteps: int, seed: int = 0):
    env = Monitor(make_env(seed=seed))
    eval_env = Monitor(make_env(seed=seed + 999))
    curve_csv = LOGS_DIR / f"{algo_name}_convergence.csv"
    if curve_csv.exists():
        curve_csv.unlink()

    class RewardCurveCallback(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            pass

    eval_log_dir = LOGS_DIR / "eval_tmp"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=None, log_path=str(eval_log_dir / algo_name),
        eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True, verbose=0,
    )

    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "timesteps")}
    model = algo_cls("MlpPolicy", env, verbose=0, seed=seed, **kwargs)

    callbacks = [eval_cb]
    if algo_name == "dqn":
        loss_csv = LOGS_DIR / "dqn_loss_curve.csv"
        if loss_csv.exists():
            loss_csv.unlink()
        callbacks.append(LossLoggerCallback(loss_csv))

    model.learn(total_timesteps=timesteps, callback=callbacks)

    # EvalCallback stores its history internally
    timesteps_arr = np.array(eval_cb.evaluations_timesteps)
    rewards_arr = np.array(eval_cb.evaluations_results).mean(axis=1)
    for t, r in zip(timesteps_arr, rewards_arr):
        log_run(curve_csv, {"timesteps": int(t), "mean_reward": round(float(r), 3)})

    env.close()
    eval_env.close()
    return model


def train_reinforce_curve(cfg: dict, seed: int = 0):
    """REINFORCE already logs a per-episode curve in pg_training.py; this
    just re-runs the baseline config cleanly and saves it under a
    predictable filename for plotting."""
    from training.pg_training import train_reinforce
    curve_target = LOGS_DIR / "reinforce_convergence.csv"
    src = LOGS_DIR / f"reinforce_curve_{cfg['name']}.csv"
    if src.exists():
        src.unlink()
    policy, mean_r, std_r, train_time = train_reinforce(cfg, seed=seed)
    if src.exists():
        src.rename(curve_target)
    return policy, mean_r


def plot_convergence():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    specs = [
        ("dqn_convergence.csv", "DQN", "timesteps"),
        ("reinforce_convergence.csv", "REINFORCE", "episode"),
        ("a2c_convergence.csv", "A2C", "timesteps"),
        ("ppo_convergence.csv", "PPO", "timesteps"),
    ]
    for ax, (fname, title, xcol) in zip(axes.flat, specs):
        path = LOGS_DIR / fname
        if not path.exists():
            ax.set_title(f"{title} (no data)")
            continue
        import pandas as pd
        df = pd.read_csv(path)
        ycol = "mean_reward" if "mean_reward" in df.columns else "episode_reward"
        ax.plot(df[xcol], df[ycol])
        ax.set_title(f"{title} convergence")
        ax.set_xlabel(xcol)
        ax.set_ylabel("reward")
    fig.tight_layout()
    ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(ASSETS_DIR / "convergence_curves.png", dpi=150)
    plt.close(fig)
    print("saved assets/convergence_curves.png")

    loss_path = LOGS_DIR / "dqn_loss_curve.csv"
    if loss_path.exists():
        import pandas as pd
        df = pd.read_csv(loss_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["timesteps"], df["loss"])
        ax.set_title("DQN training loss (objective) over time")
        ax.set_xlabel("timesteps")
        ax.set_ylabel("loss")
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / "dqn_loss_curve.png", dpi=150)
        plt.close(fig)
        print("saved assets/dqn_loss_curve.png")


def main():
    print("=== DQN convergence run ===")
    train_with_eval_curve(DQN, "dqn",
                           dict(learning_rate=1e-3, gamma=0.99, buffer_size=10000,
                                batch_size=32, exploration_fraction=0.3,
                                exploration_final_eps=0.05),
                           timesteps=30000)

    print("=== A2C convergence run ===")
    train_with_eval_curve(A2C, "a2c",
                           dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.0),
                           timesteps=30000)

    print("=== PPO convergence run ===")
    train_with_eval_curve(PPO, "ppo",
                           dict(learning_rate=3e-4, gamma=0.99, n_steps=256,
                                batch_size=64, ent_coef=0.0, clip_range=0.2),
                           timesteps=30000)

    print("=== REINFORCE convergence run ===")
    train_reinforce_curve(dict(name="reinforce_01_baseline", learning_rate=1e-3,
                                gamma=0.99, hidden_size=64, episodes=800))

    plot_convergence()


if __name__ == "__main__":
    main()
