# ArtGuard Africa 

**Mission:** ArtGuard Africa protects African artists and artisans (from
South African modernist painters to Maasai beadwork makers) from image
forgery, cloning, and AI-generated reproductions of their work.

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


## Training

```bash
# Single quick run of each algorithm (good for testing your machine)
uv run python -m training.dqn_training --timesteps 30000
uv run python -m training.pg_training --algo reinforce --episodes 800
uv run python -m training.pg_training --algo a2c --timesteps 30000
uv run python -m training.pg_training --algo ppo --timesteps 30000

# Full 10-run hyperparameter sweep for one algorithm (this is what
# produces the report's hyperparameter tables -- logs/<algo>_sweep.csv)
uv run python -m training.dqn_training --sweep
uv run python -m training.pg_training --algo reinforce --sweep
uv run python -m training.pg_training --algo a2c --sweep
uv run python -m training.pg_training --algo ppo --sweep

# After all four sweeps have run, generate report figures into assets/
uv run python -m training.plot_results
```

Each run logs one row (hyperparameters + mean/std evaluation reward +
training time) to `logs/<algo>_sweep.csv` -- that file **is** the
hyperparameter table for the report, no manual copying needed. The best
model from each sweep is saved to `models/dqn/` or `models/pg/`.

## Setup

```bash
git clone https://github.com/Muen1/Cynthia-Mutie_rl_summative
cd artguard_rl_summative
uv sync
```

## Run

```bash
# Headless smoke test (prints episode rewards)
uv run main.py --episodes 5

# With the live pygame visualization (run this on your own machine, not a
# headless server, and use this window when recording the demo video)
uv run main.py --render --episodes 1
```

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
