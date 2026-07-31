"""
Runs the BEST-PERFORMING trained agent live, with the pygame window open
and verbose terminal output explaining each decision -- this is what the
demo video should show (not main.py, which uses random actions).

Usage (default: your best DQN model, matches the recommendation in the
report's Conclusion section):
    uv run python play_best_agent.py

Run a different algorithm's best model instead:
    uv run python play_best_agent.py --algo a2c
    uv run python play_best_agent.py --algo ppo
    uv run python play_best_agent.py --algo reinforce
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import gymnasium as gym

import environment  # noqa: F401  registers ArtGuardAfrica-v0

ACTION_NAMES = {
    0: "APPROVE",
    1: "FLAG",
    2: "INVESTIGATE",
    3: "ESCALATE",
    4: "REQUEST_PROVENANCE",
}


def load_predict_fn(algo: str):
    if algo == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load("models/dqn/best_model.zip")
        return lambda obs: int(model.predict(obs, deterministic=True)[0])

    if algo in ("a2c", "ppo"):
        from stable_baselines3 import A2C, PPO
        cls = A2C if algo == "a2c" else PPO
        model = cls.load(f"models/pg/{algo}_best.zip")
        return lambda obs: int(model.predict(obs, deterministic=True)[0])

    if algo == "reinforce":
        from training.pg_training import PolicyNet
        policy = PolicyNet()
        policy.load_state_dict(torch.load("models/pg/reinforce_best.pt"))
        policy.eval()

        def predict_fn(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits = policy(obs_t)
            return int(torch.argmax(logits).item())
        return predict_fn

    raise ValueError(f"unknown algo: {algo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "a2c", "ppo", "reinforce"], default="dqn")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pause", type=float, default=0.15,
                         help="seconds to pause after each decision, so it's watchable on video")
    args = parser.parse_args()

    print(f"Loading best {args.algo.upper()} model...")
    predict_fn = load_predict_fn(args.algo)
    print("Model loaded. Launching environment...\n")

    env = gym.make("ArtGuardAfrica-v0", render_mode="human")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        env.render()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action = predict_fn(obs)
            print(
                f"Item {env.unwrapped.items_resolved + 1}/{env.unwrapped.queue_size}  "
                f"| cnn_forgery_prob={obs[0]:.2f}  embedding_similarity={obs[1]:.2f}  "
                f"seller_trust={obs[2]:.2f}  metadata_completeness={obs[4]:.2f}  "
                f"artist_risk_index={obs[5]:.2f}  "
                f"-> ACTION: {ACTION_NAMES[action]}"
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            env.render()
            time.sleep(args.pause)

        print(f"\n=== Episode {ep + 1} finished ===")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Items resolved: {info.get('items_resolved')}")
        print(f"(For reference, this model's sweep evaluation average was reported "
              f"in logs/{args.algo}_sweep.csv)\n")

    print("Closing environment window in 3 seconds...")
    time.sleep(3)
    env.close()


if __name__ == "__main__":
    main()
