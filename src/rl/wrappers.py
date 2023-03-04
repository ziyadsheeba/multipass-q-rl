import typing as t

import gym
import numpy as np
from gym.spaces import Tuple
from gym_platform.envs.platform_env import Constants


class PlatformWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """An action wrapper to scale back the normalized action parameters to the correct range.

        Args:
            env (gym.Env): The platform gym environment.
        """
        super().__init__(env)
        observation_space = self.env.observation_space
        self.low = observation_space.spaces[0].low
        self.high = observation_space.spaces[0].high
        self.observation_space = Tuple(
            (
                gym.spaces.Box(
                    low=-np.ones(self.low.shape),
                    high=np.ones(self.high.shape),
                    dtype=np.float32,
                ),
                observation_space.spaces[1],
            )
        )

    def step(
        self, action: t.Tuple[int, np.ndarray]
    ) -> t.Tuple[np.ndarray, float, bool, t.Mapping]:
        """Overrides the step function, to rescale the action parameters
        into their original range, assuming that each element was normalized in
        the range of (0,1). Also returns only the first element of the state
        tuple.

        Args:
            action (t.Tuple[int, t.Tuple[np.ndarray]]): A tuple of (discrete_action, action_parameter).

        Returns:
            t.Tuple[np.ndarray, float, bool, t.Mapping]: state, reward, done, info.
        """
        action = self._action(action)
        (obs, _), reward, done, info = self.env.step(action)
        return self.scale_state(obs), reward, done, info

    def reset(self) -> np.ndarray:
        """Overrides the reset state, to return only the firt element
        of the state tuple.

        Returns:
            np.ndarray: The environment's state.
        """
        s, _ = self.env.reset()
        return self.scale_state(s)

    def _action(
        self, action: t.Tuple[int, np.ndarray]
    ) -> t.Tuple[int, t.Tuple[np.ndarray]]:
        """Rescales the action to the correct range. Assumes
        that the continous actions are scaled in the range of (0,1) each.

        Args:
            action (t.Tuple[int, np.ndarray]): A tuple, where the first index
            contains the discrete action index, and the second index contains
            a numpy array, containing the parameterized actions for each discrete
            action.

        Returns:
            t.Tuple[int, t.Tuple[np.ndarray]]: The rescaled actions, in the format accepted by
            the environment.
        """
        action_discrete, action_parameterized = action
        action_parameterized_scaled = tuple(
            [
                np.array([(action + 1) * (act_max - act_min) / 2 + act_min])
                for act_min, act_max, action in zip(
                    Constants.PARAMETERS_MIN,
                    Constants.PARAMETERS_MAX,
                    action_parameterized,
                )
            ]
        )
        return (action_discrete, action_parameterized_scaled)

    def scale_state(self, state: np.ndarray) -> np.ndarray:
        """Scales each coordinate of the state between -1 and 1."""
        state = 2.0 * (state - self.low) / (self.high - self.low) - 1.0
        return state
