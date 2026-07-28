"""
Generalization test (rubric: "generalization tests").

Loads each algorithm's BEST saved model (from the hyperparameter sweeps)
and evaluates it on two environments:
  1. ArtGuardAfrica-v0       -- the market it was trained on
  2. ArtGuardAfrica-Hard-v0  -- a harder, distribution-shifted market
                                 (forgers have gotten better: noisier
                                 evidence, more genuinely ambiguous items)

This tells us whether each agent learned a robust decision *policy*
(evidence-weighing that still works when the evidence gets noisier) or
just overfit to the easy market's exact statistics.

Run this AFTER training/dqn_training.py --sweep and
training/pg_training.py --algo {reinforce,a2c,ppo} --sweep have produced
best_model files in models/.

Usage:
    uv run python -m training.generalization
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, A2C, PPO

from training.utils import make_env, evaluate_policy_fn, log_run, LOGS_DIR, MODELS_DIR
from training.pg_training import PolicyNet

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def make_hard_env(seed=None):
    import gymnasium as gym
    import environment  # noqa: F401
    env = gym.make("ArtGuardAfrica-Hard-v0")
    if seed is not None:
        env.reset(seed=seed)
    return env


def eval_on(env_factory, predict_fn, n_episodes=20, seed=2000):
    rewards = []
    for ep in range(n_episodes):
        env = env_factory(seed=seed + ep)
        obs, info = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action = predict_fn(obs)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
        env.close()
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    results = []
    csv_path = LOGS_DIR / "generalization_results.csv"
    if csv_path.exists():
        csv_path.unlink()

    # --- DQN ---
    dqn_path = MODELS_DIR / "dqn" / "best_model.zip"
    if dqn_path.exists():
        model = DQN.load(dqn_path)
        predict_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])
        easy_r, easy_s = eval_on(lambda seed: make_env(seed=seed), predict_fn)
        hard_r, hard_s = eval_on(make_hard_env, predict_fn)
        results.append(("DQN", easy_r, easy_s, hard_r, hard_s))
    else:
        print(f"skip DQN: {dqn_path} not found (run the sweep first)")

    # --- A2C / PPO ---
    for algo_name, algo_cls in [("A2C", A2C), ("PPO", PPO)]:
        model_path = MODELS_DIR / "pg" / f"{algo_name.lower()}_best.zip"
        if model_path.exists():
            model = algo_cls.load(model_path)
            predict_fn = lambda obs, m=model: int(m.predict(obs, deterministic=True)[0])
            easy_r, easy_s = eval_on(lambda seed: make_env(seed=seed), predict_fn)
            hard_r, hard_s = eval_on(make_hard_env, predict_fn)
            results.append((algo_name, easy_r, easy_s, hard_r, hard_s))
        else:
            print(f"skip {algo_name}: {model_path} not found (run the sweep first)")

    # --- REINFORCE ---
    reinforce_path = MODELS_DIR / "pg" / "reinforce_best.pt"
    if reinforce_path.exists():
        policy = PolicyNet()
        policy.load_state_dict(torch.load(reinforce_path))
        policy.eval()

        def predict_fn(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits = policy(obs_t)
            return int(torch.argmax(logits).item())

        easy_r, easy_s = eval_on(lambda seed: make_env(seed=seed), predict_fn)
        hard_r, hard_s = eval_on(make_hard_env, predict_fn)
        results.append(("REINFORCE", easy_r, easy_s, hard_r, hard_s))
    else:
        print(f"skip REINFORCE: {reinforce_path} not found (run the sweep first)")

    if not results:
        print("No trained models found -- run the sweeps first.")
        return

    for algo, easy_r, easy_s, hard_r, hard_s in results:
        drop = easy_r - hard_r
        drop_pct = (drop / easy_r * 100) if easy_r != 0 else float("nan")
        row = {
            "algorithm": algo,
            "easy_market_reward": round(easy_r, 2),
            "easy_market_std": round(easy_s, 2),
            "hard_market_reward": round(hard_r, 2),
            "hard_market_std": round(hard_s, 2),
            "performance_drop": round(drop, 2),
            "performance_drop_pct": round(drop_pct, 1),
        }
        log_run(csv_path, row)
        print(f"{algo}: easy={easy_r:.2f}  hard={hard_r:.2f}  drop={drop:.2f} ({drop_pct:.1f}%)")

    # plot
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["easy_market_reward"], width, label="Trained market (easy)")
    ax.bar(x + width / 2, df["hard_market_reward"], width, label="Shifted market (hard)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["algorithm"])
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title("Generalization: performance on a harder, shifted market")
    ax.legend()
    fig.tight_layout()
    ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(ASSETS_DIR / "generalization_test.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved assets/generalization_test.png and {csv_path}")


if __name__ == "__main__":
    main()
