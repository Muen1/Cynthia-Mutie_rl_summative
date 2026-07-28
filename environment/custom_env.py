"""
ArtGuard Africa — Custom Gymnasium Environment
================================================

Mission context
----------------
ArtGuard Africa protects African artists and artisans from image forgery,
cloning, and AI-generated reproductions of their work (e.g. South African
modernist paintings, Maasai beadwork). This environment simulates the
day-to-day decision problem faced by a *verification agent* sitting inside
a marketplace/gallery intake pipeline: for every artwork submitted for
listing or certification, the agent must decide what to do with it,
trading off accuracy against cost/time, before the item leaves the queue.

This is NOT a literal recreation of the CNN forgery classifier — it is a
decision-making environment for an RL agent that CONSUMES the outputs of
such a classifier (forgery probability, embedding similarity, etc.) as
part of its observation, and must learn a good *policy* for what to do
with that evidence. This keeps the environment realistic, non-trivial,
and directly traceable to production use (the trained policy could sit
behind ArtGuard Africa's intake API).

Action Space (Discrete(5))
---------------------------
0: APPROVE      — certify the artwork as authentic, list it for sale
1: FLAG         — flag the artwork as a suspected forgery, remove listing
2: INVESTIGATE  — request deeper forensic analysis (costs time, reveals
                   a more accurate observation on the NEXT step for the
                   same item before a final decision is required)
3: ESCALATE     — send to a human cultural-heritage expert (safe but slow
                   and has a real-world staffing cost)
4: REQUEST_PROVENANCE — ask the seller for provenance documents (cheap,
                   moderately informative, mirrors real vetting workflows)

Observation Space (Box, shape=(8,), float32)
---------------------------------------------
0: cnn_forgery_prob        — [0,1] forgery classifier's raw score
1: embedding_similarity    — [0,1] similarity to nearest authenticated ref work
2: seller_trust_score      — [0,1] marketplace seller reputation
3: price_deviation         — [-1,1] how far listed price deviates from
                              fair market price for that artist/style
4: metadata_completeness   — [0,1] fraction of expected provenance metadata present
5: artist_risk_index       — [0,1] how frequently this artist/style is targeted
                              (modelled on the ~30% forgery-rate research finding)
6: queue_pressure          — [0,1] how full the day's intake queue is (time pressure)
7: investigation_bonus     — [0,1] extra evidence revealed after an INVESTIGATE action
                              (0 until the agent has investigated this item)

Rewards
-------
Ground truth for each item (authentic / forged) is hidden from the agent
and only used to compute reward — mirroring real deployment, where the
agent never gets to see the "answer" directly, only noisy evidence.

  Correct APPROVE  (item authentic)              : +5
  Correct FLAG     (item forged)                 : +6   (protecting a livelihood
                                                          matters slightly more
                                                          than a smooth sale)
  False APPROVE    (forged item let through)      : -10  (major mission failure)
  False FLAG       (authentic item blocked)       : -6   (reputational/economic harm
                                                          to a real artist)
  ESCALATE                                        : -1   (staffing cost) but +3 if
                                                          it was a genuinely hard/
                                                          ambiguous case (reduces risk)
  REQUEST_PROVENANCE                              : -0.5 cost, reveals more metadata
  INVESTIGATE                                     : -1.5 cost (time), but grants a
                                                          better observation next step
  Any step                                        : -0.1 living cost (encourages
                                                          decisive, efficient policies)

Start State
-----------
A new episode represents one day's intake queue. Each item is freshly
sampled: forgery ground truth is drawn with probability = artist_risk_index
(itself sampled per item, centered near the ~30% real-world figure), and
all observation features are generated consistently with that ground
truth plus noise.

Terminal Conditions
--------------------
- Episode ends after `queue_size` items have been finally resolved
  (APPROVE / FLAG / ESCALATE all resolve an item; INVESTIGATE and
  REQUEST_PROVENANCE do not resolve it and consume a step without
  advancing the queue, up to a max of 2 extra looks per item).
- Truncated if `max_steps` is exceeded (safety net).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


APPROVE, FLAG, INVESTIGATE, ESCALATE, REQUEST_PROVENANCE = range(5)
ACTION_NAMES = {
    APPROVE: "APPROVE",
    FLAG: "FLAG",
    INVESTIGATE: "INVESTIGATE",
    ESCALATE: "ESCALATE",
    REQUEST_PROVENANCE: "REQUEST_PROVENANCE",
}


class ArtAuthenticationEnv(gym.Env):
    """RL environment simulating ArtGuard Africa's verification agent."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, queue_size: int = 20, max_extra_looks: int = 2,
                 render_mode: str | None = None, seed: int | None = None,
                 risk_mean: float = 0.30, ambiguous_prob: float = 0.25,
                 evidence_noise: float = 0.15):
        """
        risk_mean / ambiguous_prob / evidence_noise let us construct a
        harder, distribution-shifted variant of the market (see
        ArtGuardAfrica-Hard-v0 in __init__.py) for generalization testing:
        a future market where forgers have gotten better (evidence is
        noisier and more items are genuinely ambiguous) than what the
        agent trained on.
        """
        super().__init__()
        self.queue_size = queue_size
        self.max_extra_looks = max_extra_looks
        self.render_mode = render_mode
        self.risk_mean = risk_mean
        self.ambiguous_prob = ambiguous_prob
        self.evidence_noise = evidence_noise

        self.action_space = spaces.Discrete(5)
        low = np.array([0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._pygame_ready = False

        # episode state
        self.items_resolved = 0
        self.step_count = 0
        self.looks_this_item = 0
        self.current_item = None  # dict with ground truth + obs
        self.history = []  # for rendering: list of (obs, action, correct)

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.items_resolved = 0
        self.step_count = 0
        self.looks_this_item = 0
        self.history = []
        self.current_item = self._sample_item()
        obs = self._make_observation()
        info = {"is_forged": self.current_item["is_forged"]}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.step_count += 1
        reward = -0.1  # living cost
        terminated = False
        truncated = False
        queue_pressure = self.items_resolved / self.queue_size

        is_forged = self.current_item["is_forged"]
        resolved = True
        correct = None

        if action == APPROVE:
            reward += 5.0 if not is_forged else -10.0
            correct = not is_forged
        elif action == FLAG:
            reward += 6.0 if is_forged else -6.0
            correct = is_forged
        elif action == ESCALATE:
            hard_case = self.current_item["ambiguous"]
            reward += (3.0 if hard_case else -1.0)
            correct = True  # escalation always ends up correct (human review)
        elif action == INVESTIGATE:
            resolved = False
            reward -= 1.5
            if self.looks_this_item < self.max_extra_looks:
                self.current_item["investigation_bonus"] = min(
                    1.0, self.current_item["investigation_bonus"] + 0.5
                )
                self.looks_this_item += 1
            else:
                resolved = True  # force resolution if agent stalls too long
                correct = None
        elif action == REQUEST_PROVENANCE:
            resolved = False
            reward -= 0.5
            if self.looks_this_item < self.max_extra_looks:
                self.current_item["metadata_completeness"] = min(
                    1.0, self.current_item["metadata_completeness"] + 0.4
                )
                self.looks_this_item += 1
            else:
                resolved = True
                correct = None

        self.history.append((dict(self.current_item), action, correct))

        if resolved:
            self.items_resolved += 1
            self.looks_this_item = 0
            if self.items_resolved >= self.queue_size:
                terminated = True
            else:
                self.current_item = self._sample_item()

        if self.step_count >= self.queue_size * (self.max_extra_looks + 2):
            truncated = True

        obs = self._make_observation(queue_pressure=queue_pressure)
        info = {
            "is_forged": self.current_item["is_forged"] if not terminated else None,
            "correct": correct,
            "items_resolved": self.items_resolved,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_item(self) -> dict:
        rng = self._rng
        artist_risk_index = float(np.clip(rng.normal(self.risk_mean, 0.12), 0.02, 0.9))
        is_forged = rng.random() < artist_risk_index
        ambiguous = rng.random() < self.ambiguous_prob  # genuinely hard case, benefits from escalation
        noise = self.evidence_noise

        if is_forged:
            cnn_forgery_prob = float(np.clip(rng.normal(0.72, noise), 0, 1))
            embedding_similarity = float(np.clip(rng.normal(0.55, noise), 0, 1))
            metadata_completeness = float(np.clip(rng.normal(0.35, noise), 0, 1))
        else:
            cnn_forgery_prob = float(np.clip(rng.normal(0.25, noise), 0, 1))
            embedding_similarity = float(np.clip(rng.normal(0.85, noise * 0.67), 0, 1))
            metadata_completeness = float(np.clip(rng.normal(0.7, noise), 0, 1))

        if ambiguous:
            # push evidence toward the noisy middle regardless of ground truth
            cnn_forgery_prob = float(np.clip(cnn_forgery_prob * 0.5 + 0.25, 0, 1))
            embedding_similarity = float(np.clip(embedding_similarity * 0.5 + 0.35, 0, 1))

        return {
            "is_forged": is_forged,
            "ambiguous": ambiguous,
            "cnn_forgery_prob": cnn_forgery_prob,
            "embedding_similarity": embedding_similarity,
            "seller_trust_score": float(np.clip(rng.normal(0.6, 0.2), 0, 1)),
            "price_deviation": float(np.clip(rng.normal(0.0, 0.4), -1, 1)),
            "metadata_completeness": metadata_completeness,
            "artist_risk_index": artist_risk_index,
            "investigation_bonus": 0.0,
        }

    def _make_observation(self, queue_pressure: float | None = None) -> np.ndarray:
        item = self.current_item
        if queue_pressure is None:
            queue_pressure = self.items_resolved / self.queue_size
        obs = np.array([
            item["cnn_forgery_prob"],
            item["embedding_similarity"],
            item["seller_trust_score"],
            item["price_deviation"],
            item["metadata_completeness"],
            item["artist_risk_index"],
            queue_pressure,
            item["investigation_bonus"],
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    # Rendering (see environment/rendering.py for the pygame implementation)
    # ------------------------------------------------------------------
    def render(self):
        from environment.rendering import render_frame
        return render_frame(self)

    def close(self):
        from environment.rendering import close_renderer
        close_renderer()
