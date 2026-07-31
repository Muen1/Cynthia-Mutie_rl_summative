# ArtGuard Africa 

**Mission:** African art is vulnerable to mass-produced imitations, a problem that is rapidly intensifying. Academic research from the University of Pretoria found that a search of the work of South African modernist painter Lucky Sibiya returns results in which roughly 30% of the images are forgeries, with at least 21 named black South African modernist artists from the 1960–1990 period, including George Pemba, Dumile Feni, Julian Motau, and Welcome Koboka, confirmed as forgery targets. That is why ArtGuard Africa aims to protect African artists from image forgery since forgery detection tools were built for western art traditions and exclude African artists entirely. So instead of building the CNN forgery classifier itself, I build a reinforcement learning agent that compares four algorithms (DQN, REINFORCE, A2C, and PPO)

This repository trains and compares four reinforcement learning agents
(DQN, REINFORCE, A2C, PPO) on a custom environment, `ArtGuardAfrica-v0`,
that simulates the decision problem faced by a **verification agent**
inside an art-marketplace intake pipeline: given evidence about a
submitted artwork (a forgery-classifier score, embedding similarity to
known authentic works, seller trust, metadata completeness, etc.), the
agent must decide to **approve**, **flag as forgery**, **investigate
further**, **escalate to a human expert**, or **request provenance
documents** — before the item leaves the queue.

Full environment spec (actions, observations, rewards, start/terminal
conditions) is documented at the top of `environment/custom_env.py`.

## Video demo

[YouTube Link](https://youtu.be/KR2DHNgUCds)

## Setup

```bash
git clone https://github.com/Muen1/Cynthia-Mutie_rl_summative
cd Cynthia-Mutie_rl_summative
uv sync
```

## Run the best performing agent 

```bash
uv run python play_best_agent.py --algo dqn --pause 0.9
```

## Quick environment demo (random actions for testing only)

```bash
uv run main.py --episodes 5              # headless smoke test
uv run main.py --render --episodes 1      # with the pygame window
```
## Training

```bash
# Single quick run of each algorithm
uv run python -m training.dqn_training --timesteps 30000
uv run python -m training.pg_training --algo reinforce --episodes 800
uv run python -m training.pg_training --algo a2c --timesteps 30000
uv run python -m training.pg_training --algo ppo --timesteps 30000

# Full 10-run hyperparameter sweep per algorithm (produces the report's
# hyperparameter tables -- logs/<algo>_sweep.csv)
uv run python -m training.dqn_training --sweep
uv run python -m training.pg_training --algo reinforce --sweep
uv run python -m training.pg_training --algo a2c --sweep
uv run python -m training.pg_training --algo ppo --sweep
```

## Analysis (convergence, training stability, generalization)

```bash
# Reward-over-training-progress curves + DQN loss curve
uv run python -m training.convergence

# Evaluate each best model on a harder, distribution-shifted market
uv run python -m training.generalization

# Generate all report figures into assets/ from the sweep CSVs
uv run python -m training.plot_results
```

## Tests

```bash
uv run pytest tests/ -q
```

## Results Summary

| Algorithm | Best Mean Reward (Sweep) | Generalization Drop (Harder Market) |
|-----------|--------------------------:|------------------------------------:|
| DQN | **100.75** | 21.4% |
| A2C | 99.95 | **16.9%** |
| PPO | 100.20 | 22.4% |
| REINFORCE | 100.25 | 30.8% |

### Recommended Model

**DQN** is the recommended model for deployment because it provides:

- Highest mean reward during hyperparameter sweeps (**100.75**)
- Fastest convergence during training
- Strong generalization to unseen market conditions
- Deterministic action selection, making decisions fully auditable (the same input always produces the same action)

> While **A2C** achieved the smallest performance drop under harder market conditions, **DQN** offered the best overall balance of reward, convergence speed, robustness, and deployment practicality.

For a complete evaluation, methodology, and discussion, see the accompanying report.

## Repository layout

```
project_root/
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py
├── environment/
│   ├── __init__.py       # registers ArtGuardAfrica-v0 with Gymnasium
│   ├── custom_env.py      # the environment
│   └── rendering.py       # pygame visualization
├── training/               # DQN + policy-gradient training scripts
├── models/                 # saved trained models
├── logs/                   # training logs for tensorboard/plots
├── assets/                  # report images / recorded gifs
└── tests/                   # sanity tests
```
