"""Shared helpers used by every training script."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import environment  # noqa: F401  registers ArtGuardAfrica-v0

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def make_env(seed: int | None = None):
    env = gym.make("ArtGuardAfrica-v0")
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate_policy_fn(predict_fn, n_episodes: int = 20, seed: int = 1000):
    """Run n_episodes with a callable predict_fn(obs) -> action and return
    (mean_reward, std_reward) — used identically for every algorithm so
    comparisons across DQN/REINFORCE/A2C/PPO are apples-to-apples."""
    env = make_env()
    rewards = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action = predict_fn(obs)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def log_run(csv_path: Path, row: dict):
    """Append one hyperparameter-run result row to a CSV table (creates the
    file with a header the first time). This CSV becomes the hyperparameter
    table required in the report — one row per run."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def timer():
    return time.time()
