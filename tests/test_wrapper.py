import gym
import gym_platform
from gym_platform.envs.platform_env import Constants

from rl.wrappers import PlatformWrapper
import numpy as np


def test_wrapper() -> None:
    """Tests that the wrapped and the unwrapped environments behave the same way.
    """
    env = gym.make("Platform-v0")
    env.seed(0)
    env_wrapped = PlatformWrapper(gym.make("Platform-v0"))
    env_wrapped.seed(0)
    s_wrapped = env_wrapped.reset()
    action = env.action_space.sample()
    action_discrete = action[0]
    action_continous = 2*(np.concatenate(action[1]) - Constants.PARAMETERS_MIN)/(Constants.PARAMETERS_MAX - Constants.PARAMETERS_MIN) - 1
    s_prime_wrapped, reward_wrapped, _, _ = env_wrapped.step(action=(action_discrete, action_continous))
    s, _ = env.reset()
    (s_prime,_), reward, _, _ = env.step(action=action)
    assert np.allclose(s_prime_wrapped, 2*s_prime - 1)
    assert np.allclose(s_wrapped, 2*s -1)
    assert reward_wrapped == reward
    
