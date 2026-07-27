"""
Generates the comparison plots required by the "Discussion & Analysis"
rubric item, reading from the CSVs that dqn_training.py / pg_training.py
produce in logs/. Run this AFTER the full hyperparameter sweeps (Phase 3).

Usage:
    uv run training/plot_results.py
Outputs go to assets/ (ready to drop straight into the report).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def plot_sweep_bar(csv_name: str, title: str, out_name: str):
    path = LOGS_DIR / csv_name
    if not path.exists():
        print(f"skip: {path} not found yet")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["name"], df["mean_reward"], yerr=df["std_reward"], capsize=3)
    ax.set_title(title)
    ax.set_ylabel("Mean evaluation reward")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(ASSETS_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"saved {out_name}")


def plot_algorithm_comparison():
    """One subplot figure comparing best run of each algorithm — the
    'cumulative reward curves (all methods in subplots)' the rubric asks for."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    files_titles = [
        ("dqn_sweep.csv", "DQN"),
        ("reinforce_sweep.csv", "REINFORCE"),
        ("a2c_sweep.csv", "A2C"),
        ("ppo_sweep.csv", "PPO"),
    ]
    for ax, (fname, title) in zip(axes.flat, files_titles):
        path = LOGS_DIR / fname
        if not path.exists():
            ax.set_title(f"{title} (no data yet)")
            continue
        df = pd.read_csv(path)
        ax.plot(df["mean_reward"].values, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Run index")
        ax.set_ylabel("Mean reward")
    fig.tight_layout()
    ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(ASSETS_DIR / "algorithm_comparison.png", dpi=150)
    plt.close(fig)
    print("saved algorithm_comparison.png")


def plot_reinforce_entropy():
    """PG entropy curve — reads the per-episode curve logged during the
    baseline REINFORCE run."""
    path = LOGS_DIR / "reinforce_curve_reinforce_01_baseline.csv"
    if not path.exists():
        print(f"skip: {path} not found yet (run the reinforce sweep first)")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["episode"], df["mean_entropy"])
    ax.set_title("REINFORCE policy entropy over training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean entropy")
    fig.tight_layout()
    ASSETS_DIR.mkdir(exist_ok=True)
    fig.savefig(ASSETS_DIR / "reinforce_entropy.png", dpi=150)
    plt.close(fig)
    print("saved reinforce_entropy.png")


if __name__ == "__main__":
    plot_sweep_bar("dqn_sweep.csv", "DQN — reward by hyperparameter run", "dqn_sweep.png")
    plot_sweep_bar("a2c_sweep.csv", "A2C — reward by hyperparameter run", "a2c_sweep.png")
    plot_sweep_bar("ppo_sweep.csv", "PPO — reward by hyperparameter run", "ppo_sweep.png")
    plot_sweep_bar("reinforce_sweep.csv", "REINFORCE — reward by hyperparameter run", "reinforce_sweep.png")
    plot_algorithm_comparison()
    plot_reinforce_entropy()
