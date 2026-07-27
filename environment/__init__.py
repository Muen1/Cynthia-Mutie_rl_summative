from gymnasium.envs.registration import register
from environment.custom_env import ArtAuthenticationEnv

register(
    id="ArtGuardAfrica-v0",
    entry_point="environment.custom_env:ArtAuthenticationEnv",
    max_episode_steps=200,
)

__all__ = ["ArtAuthenticationEnv"]
