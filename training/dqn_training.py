"""
DQN training for ArtGuardAfrica-v0 (value-based method).

Usage
-----
Single run, default hyperparameters:
    uv run training/dqn_training.py

Full hyperparameter sweep (>=10 combinations, appends to logs/dqn_sweep.csv,
best model saved to models/dqn/best_model.zip):
    uv run training/dqn_training.py --sweep

Each run trains fresh, evaluates on 20 held-out episodes using the SAME
evaluation function as every other algorithm (training/utils.py), and logs
one row per run to logs/dqn_sweep.csv — that row becomes one line of the
report's DQN hyperparameter table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from training.utils import make_env, evaluate_policy_fn, log_run, timer, LOGS_DIR, MODELS_DIR

MODEL_DIR = MODELS_DIR / "dqn"

# Ten distinct hyperparameter combinations covering the knobs the rubric
# explicitly calls out for DQN: learning_rate, gamma, buffer_size,
# exploration (exploration_fraction / exploration_final_eps), batch_size.
SWEEP_CONFIGS = [
    dict(name="dqn_01_baseline",      learning_rate=1e-3, gamma=0.99, buffer_size=10000,
         batch_size=32,  exploration_fraction=0.3, exploration_final_eps=0.05, timesteps=30000),
    dict(name="dqn_02_low_lr",        learning_rate=1e-4, gamma=0.99, buffer_size=10000,
         batch_size=32,  exploration_fraction=0.3, exploration_final_eps=0.05, timesteps=30000),
    dict(name="dqn_03_high_lr",       learning_rate=5e-3, gamma=0.99, buffer_size=10000,
         batch_size=32,  exploration_fraction=0.3, exploration_final_eps=0.05, timesteps=30000),
    dict(name="dqn_04_low_gamma",     learning_rate=1e-3, gamma=0.90, buffer_size=10000,
         batch_size=32,  exploration_fraction=0.3, exploration_final_eps=0.05, timesteps=30000),
    dict(name="dqn_05_high_gamma",    learning_rate=1e-3, gamma=0.999, buffer_size=10000,
         batch_size=32,  exploration_fraction=0.3, exploration_final_eps=0.05, timesteps=30000),
    dict(name="dqn_06_small_buffer",  learning_rate=1e-3, gamma=0.99, buffer_size=1000,
         batch_size=32,  exploration_fraction=0.3, exploration_final_eps=0.05, timesteps=30000),
    dict(name="dqn_07_large_buffer",  learning_rate=1e-3, gamma=0.99, buffer_size=50000,
         batch_size=32,  exploration_fraction=0.3, exploration_final_eps=0.05, timesteps=30000),
    dict(name="dqn_08_more_explore",  learning_rate=1e-3, gamma=0.99, buffer_size=10000,
         batch_size=32,  exploration_fraction=0.6, exploration_final_eps=0.10, timesteps=30000),
    dict(name="dqn_09_less_explore",  learning_rate=1e-3, gamma=0.99, buffer_size=10000,
         batch_size=32,  exploration_fraction=0.1, exploration_final_eps=0.01, timesteps=30000),
    dict(name="dqn_10_big_batch",     learning_rate=1e-3, gamma=0.99, buffer_size=10000,
         batch_size=128, exploration_fraction=0.3, exploration_final_eps=0.05, timesteps=30000),
]


def train_one(cfg: dict, seed: int = 0, tb_log: bool = True):
    env = Monitor(make_env(seed=seed))
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_final_eps=cfg["exploration_final_eps"],
        verbose=0,
        seed=seed,
        tensorboard_log=str(LOGS_DIR / "tensorboard" / "dqn") if tb_log else None,
    )
    t0 = timer()
    model.learn(total_timesteps=cfg["timesteps"], tb_log_name=cfg["name"])
    train_time = timer() - t0

    def predict_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    mean_r, std_r = evaluate_policy_fn(predict_fn)
    env.close()
    return model, mean_r, std_r, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="run all 10 hyperparameter configs")
    parser.add_argument("--timesteps", type=int, default=30000, help="used only for a single run")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOGS_DIR / "dqn_sweep.csv"

    configs = SWEEP_CONFIGS if args.sweep else [
        dict(name="dqn_manual_run", learning_rate=1e-3, gamma=0.99, buffer_size=10000,
             batch_size=32, exploration_fraction=0.3, exploration_final_eps=0.05,
             timesteps=args.timesteps)
    ]

    best_mean = -1e9
    for cfg in configs:
        print(f"\n=== Training {cfg['name']} ===")
        model, mean_r, std_r, train_time = train_one(cfg)
        print(f"{cfg['name']}: mean_reward={mean_r:.2f} +/- {std_r:.2f}  "
              f"({train_time:.1f}s, {cfg['timesteps']} steps)")

        row = {**cfg, "mean_reward": round(mean_r, 3), "std_reward": round(std_r, 3),
               "train_time_s": round(train_time, 1)}
        log_run(csv_path, row)

        if mean_r > best_mean:
            best_mean = mean_r
            model.save(MODEL_DIR / "best_model")
            with open(MODEL_DIR / "best_model_config.json", "w") as f:
                json.dump({**cfg, "mean_reward": mean_r}, f, indent=2)

    print(f"\nBest DQN run: mean_reward={best_mean:.2f} -> saved to {MODEL_DIR/'best_model.zip'}")
    print(f"All runs logged to {csv_path}")


if __name__ == "__main__":
    main()
