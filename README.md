# ArtGuard Africa — RL Summative

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

## Project status

- [x] **Phase 1 — Environment**: custom Gymnasium env + pygame
      visualization, validated with `gymnasium.utils.env_checker`.
- [ ] **Phase 2 — Training**: DQN / REINFORCE / A2C / PPO training
      scripts, hyperparameter sweep tables, comparison plots.
- [ ] **Phase 3 — Video + Report**.

## Setup (uv only — no manual venv needed)

```bash
git clone https://github.com/Muen1/Cynthia-Mutie_rl_summative
cd Cynthia-Mutie_rl_summative
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
├── training/               # (Phase 2) DQN + policy-gradient training scripts
├── models/                 # (Phase 2) saved trained models
├── logs/                   # (Phase 2) training logs for tensorboard/plots
├── assets/                  # report images / recorded gifs
└── tests/                   # sanity tests
```
