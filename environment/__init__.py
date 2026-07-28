from gymnasium.envs.registration import register
from environment.custom_env import ArtAuthenticationEnv

register(
    id="ArtGuardAfrica-v0",
    entry_point="environment.custom_env:ArtAuthenticationEnv",
    max_episode_steps=200,
)

# Harder, distribution-shifted market used for generalization testing:
# forgers have improved (noisier evidence) and more items are genuinely
# ambiguous than what the agent trained on.
register(
    id="ArtGuardAfrica-Hard-v0",
    entry_point="environment.custom_env:ArtAuthenticationEnv",
    max_episode_steps=200,
    kwargs=dict(risk_mean=0.45, ambiguous_prob=0.45, evidence_noise=0.28),
)

__all__ = ["ArtAuthenticationEnv"]
