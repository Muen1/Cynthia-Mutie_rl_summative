"""
Quick manual test / demo runner for the ArtGuard Africa environment.

Usage:
    uv run main.py            # random-agent smoke test, no window
    uv run main.py --render   # opens a pygame window and shows the agent live
"""
import argparse
import numpy as np
import gymnasium as gym
import environment  # noqa: F401  (registers ArtGuardAfrica-v0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="show pygame window")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = gym.make("ArtGuardAfrica-v0", render_mode=render_mode)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if args.render:
                env.render()
        print(f"Episode {ep + 1}: steps={steps} total_reward={total_reward:.2f} "
              f"items_resolved={info.get('items_resolved')}")

    env.close()


if __name__ == "__main__":
    main()
