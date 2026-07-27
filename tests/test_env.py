import gymnasium as gym
import environment  # noqa: F401


def test_env_registers_and_resets():
    env = gym.make("ArtGuardAfrica-v0")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_env_runs_full_episode():
    env = gym.make("ArtGuardAfrica-v0")
    obs, info = env.reset(seed=0)
    done = False
    steps = 0
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done


def test_heuristic_beats_random():
    """A simple 'trust the CNN score' heuristic should score much higher
    than random actions, confirming the reward signal is learnable."""
    import numpy as np

    def run(policy_fn, seed):
        env = gym.make("ArtGuardAfrica-v0")
        obs, info = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            action = policy_fn(obs)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
        return total

    heuristic = lambda obs: 1 if obs[0] > 0.5 else 0
    random_policy = lambda obs: np.random.randint(0, 5)

    heuristic_score = run(heuristic, seed=1)
    random_score = run(random_policy, seed=1)
    assert heuristic_score > random_score
