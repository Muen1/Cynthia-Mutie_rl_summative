"""
Policy-gradient training for ArtGuardAfrica-v0.

Handles all three policy-gradient methods required by the assignment:
  --algo reinforce   custom REINFORCE (stable-baselines3 has no REINFORCE,
                      so this is a small hand-written PyTorch implementation)
  --algo a2c         stable-baselines3 A2C  (Actor-Critic)
  --algo ppo         stable-baselines3 PPO  (Proximal Policy Optimization)

Usage
-----
    uv run training/pg_training.py --algo reinforce
    uv run training/pg_training.py --algo a2c --sweep
    uv run training/pg_training.py --algo ppo --sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor

from training.utils import make_env, evaluate_policy_fn, log_run, timer, LOGS_DIR, MODELS_DIR

MODEL_DIR = MODELS_DIR / "pg"


# ---------------------------------------------------------------------------
# REINFORCE (hand-written — stable-baselines3 does not include it)
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=8, n_actions=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # logits


def train_reinforce(cfg: dict, seed: int = 0):
    torch.manual_seed(seed)
    env = make_env(seed=seed)
    policy = PolicyNet(hidden=cfg["hidden_size"])
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])

    csv_path = LOGS_DIR / f"reinforce_curve_{cfg['name']}.csv"
    t0 = timer()

    for episode in range(cfg["episodes"]):
        obs, info = env.reset(seed=seed + episode)
        log_probs, rewards, entropies = [], [], []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            obs, reward, term, trunc, info = env.step(int(action.item()))
            rewards.append(reward)
            done = term or trunc

        # discounted returns, normalized (variance reduction)
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + cfg["gamma"] * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -torch.stack([lp * G for lp, G in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0 or episode == cfg["episodes"] - 1:
            mean_entropy = torch.stack(entropies).mean().item()
            log_run(csv_path, {
                "episode": episode,
                "episode_reward": round(sum(rewards), 3),
                "loss": round(loss.item(), 4),
                "mean_entropy": round(mean_entropy, 4),
            })

    train_time = timer() - t0
    env.close()

    def predict_fn(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits = policy(obs_t)
        return int(torch.argmax(logits).item())

    mean_r, std_r = evaluate_policy_fn(predict_fn)
    return policy, mean_r, std_r, train_time


REINFORCE_SWEEP = [
    dict(name="reinforce_01_baseline", learning_rate=1e-3, gamma=0.99, hidden_size=64, episodes=800),
    dict(name="reinforce_02_low_lr",   learning_rate=1e-4, gamma=0.99, hidden_size=64, episodes=800),
    dict(name="reinforce_03_high_lr",  learning_rate=5e-3, gamma=0.99, hidden_size=64, episodes=800),
    dict(name="reinforce_04_low_gamma",learning_rate=1e-3, gamma=0.90, hidden_size=64, episodes=800),
    dict(name="reinforce_05_high_gamma",learning_rate=1e-3, gamma=0.999, hidden_size=64, episodes=800),
    dict(name="reinforce_06_small_net",learning_rate=1e-3, gamma=0.99, hidden_size=16, episodes=800),
    dict(name="reinforce_07_big_net",  learning_rate=1e-3, gamma=0.99, hidden_size=128, episodes=800),
    dict(name="reinforce_08_short",    learning_rate=1e-3, gamma=0.99, hidden_size=64, episodes=300),
    dict(name="reinforce_09_long",     learning_rate=1e-3, gamma=0.99, hidden_size=64, episodes=1500),
    dict(name="reinforce_10_high_lr_big_net", learning_rate=3e-3, gamma=0.99, hidden_size=128, episodes=800),
]


# ---------------------------------------------------------------------------
# A2C / PPO (stable-baselines3)
# ---------------------------------------------------------------------------
A2C_SWEEP = [
    dict(name="a2c_01_baseline",   learning_rate=7e-4, gamma=0.99, n_steps=5,  ent_coef=0.0,  timesteps=30000),
    dict(name="a2c_02_low_lr",     learning_rate=1e-4, gamma=0.99, n_steps=5,  ent_coef=0.0,  timesteps=30000),
    dict(name="a2c_03_high_lr",    learning_rate=3e-3, gamma=0.99, n_steps=5,  ent_coef=0.0,  timesteps=30000),
    dict(name="a2c_04_low_gamma",  learning_rate=7e-4, gamma=0.90, n_steps=5,  ent_coef=0.0,  timesteps=30000),
    dict(name="a2c_05_high_gamma", learning_rate=7e-4, gamma=0.999,n_steps=5,  ent_coef=0.0,  timesteps=30000),
    dict(name="a2c_06_more_steps", learning_rate=7e-4, gamma=0.99, n_steps=20, ent_coef=0.0,  timesteps=30000),
    dict(name="a2c_07_entropy",    learning_rate=7e-4, gamma=0.99, n_steps=5,  ent_coef=0.01, timesteps=30000),
    dict(name="a2c_08_high_entropy",learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.05, timesteps=30000),
    dict(name="a2c_09_few_steps",  learning_rate=7e-4, gamma=0.99, n_steps=2,  ent_coef=0.0,  timesteps=30000),
    dict(name="a2c_10_tuned",      learning_rate=1e-3, gamma=0.97, n_steps=10, ent_coef=0.01, timesteps=30000),
]

PPO_SWEEP = [
    dict(name="ppo_01_baseline",   learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, ent_coef=0.0,  clip_range=0.2, timesteps=30000),
    dict(name="ppo_02_low_lr",     learning_rate=1e-4, gamma=0.99, n_steps=256, batch_size=64, ent_coef=0.0,  clip_range=0.2, timesteps=30000),
    dict(name="ppo_03_high_lr",    learning_rate=1e-3, gamma=0.99, n_steps=256, batch_size=64, ent_coef=0.0,  clip_range=0.2, timesteps=30000),
    dict(name="ppo_04_low_gamma",  learning_rate=3e-4, gamma=0.90, n_steps=256, batch_size=64, ent_coef=0.0,  clip_range=0.2, timesteps=30000),
    dict(name="ppo_05_high_gamma", learning_rate=3e-4, gamma=0.999,n_steps=256, batch_size=64, ent_coef=0.0,  clip_range=0.2, timesteps=30000),
    dict(name="ppo_06_small_batch",learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=16, ent_coef=0.0,  clip_range=0.2, timesteps=30000),
    dict(name="ppo_07_entropy",    learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, ent_coef=0.01, clip_range=0.2, timesteps=30000),
    dict(name="ppo_08_wide_clip",  learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, ent_coef=0.0,  clip_range=0.4, timesteps=30000),
    dict(name="ppo_09_narrow_clip",learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, ent_coef=0.0,  clip_range=0.1, timesteps=30000),
    dict(name="ppo_10_more_steps", learning_rate=3e-4, gamma=0.99, n_steps=1024,batch_size=128,ent_coef=0.01, clip_range=0.2, timesteps=30000),
]


def train_sb3(algo_cls, cfg: dict, algo_name: str, seed: int = 0):
    env = Monitor(make_env(seed=seed))
    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "timesteps")}
    model = algo_cls(
        "MlpPolicy", env, verbose=0, seed=seed,
        tensorboard_log=str(LOGS_DIR / "tensorboard" / algo_name),
        **kwargs,
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
    parser.add_argument("--algo", choices=["reinforce", "a2c", "ppo"], required=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--episodes", type=int, default=800, help="REINFORCE only")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOGS_DIR / f"{args.algo}_sweep.csv"
    best_mean = -1e9

    if args.algo == "reinforce":
        configs = REINFORCE_SWEEP if args.sweep else [
            dict(name="reinforce_manual_run", learning_rate=1e-3, gamma=0.99,
                 hidden_size=64, episodes=args.episodes)
        ]
        for cfg in configs:
            print(f"\n=== Training {cfg['name']} ===")
            policy, mean_r, std_r, train_time = train_reinforce(cfg)
            print(f"{cfg['name']}: mean_reward={mean_r:.2f} +/- {std_r:.2f} ({train_time:.1f}s)")
            row = {**cfg, "mean_reward": round(mean_r, 3), "std_reward": round(std_r, 3),
                   "train_time_s": round(train_time, 1)}
            log_run(csv_path, row)
            if mean_r > best_mean:
                best_mean = mean_r
                torch.save(policy.state_dict(), MODEL_DIR / "reinforce_best.pt")
                with open(MODEL_DIR / "reinforce_best_config.json", "w") as f:
                    json.dump({**cfg, "mean_reward": mean_r}, f, indent=2)

    else:
        algo_cls = {"a2c": A2C, "ppo": PPO}[args.algo]
        default_sweep = {"a2c": A2C_SWEEP, "ppo": PPO_SWEEP}[args.algo]
        configs = default_sweep if args.sweep else [default_sweep[0] | {"timesteps": args.timesteps,
                                                                          "name": f"{args.algo}_manual_run"}]
        for cfg in configs:
            print(f"\n=== Training {cfg['name']} ===")
            model, mean_r, std_r, train_time = train_sb3(algo_cls, cfg, args.algo)
            print(f"{cfg['name']}: mean_reward={mean_r:.2f} +/- {std_r:.2f} ({train_time:.1f}s)")
            row = {**cfg, "mean_reward": round(mean_r, 3), "std_reward": round(std_r, 3),
                   "train_time_s": round(train_time, 1)}
            log_run(csv_path, row)
            if mean_r > best_mean:
                best_mean = mean_r
                model.save(MODEL_DIR / f"{args.algo}_best")
                with open(MODEL_DIR / f"{args.algo}_best_config.json", "w") as f:
                    json.dump({**cfg, "mean_reward": mean_r}, f, indent=2)

    print(f"\nBest {args.algo} run: mean_reward={best_mean:.2f}")
    print(f"All runs logged to {csv_path}")


if __name__ == "__main__":
    main()
