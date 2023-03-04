import typing as t

import numpy as np

from rl.memory import Transition


def pad_action(
    action: t.Tuple[int, np.ndarray],
) -> np.ndarray:
    """Pads the parameterized action with zeros."""
    actions_padded = np.zeros((len(action[1]),))
    actions_padded[action[0]] = action[1][action[0]]
    return actions_padded


def create_transition(
    state: np.ndarray,
    action: t.Tuple[int, np.ndarray],
    reward: float,
    next_state: np.ndarray,
    done: bool,
    next_action: t.Optional[t.Tuple[int, t.Tuple[np.ndarray]]] = None,
) -> Transition:
    """Creates a transition from what the environment returns.

    Args:
        state (np.ndarray): The current state.
        action (t.Tuple[int, np.ndarray]): The played action.
        reward (float): The reward recieved.
        next_state (np.ndarray): The next state.
        next_action (t.Optional[t.Tuple[int, t.Tuple[np.ndarray]]]): The next action played from next_state. Defaults to None.
    Returns:
        Transition: A transition object.
    """
    transition = Transition(
        state=state,
        action_discrete=action[0],
        action_continous=pad_action(action),
        reward=reward,
        next_state=next_state,
        done=done,
        next_action_discrete=next_action[0] if next_action is not None else None,
        next_action_continous=next_action[1][next_action[0]].item()
        if next_action is not None
        else None,
    )
    return transition
