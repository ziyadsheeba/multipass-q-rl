import logging
import random
import typing as t
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Transition:
    """Defines the transition elements to be stored in memory."""

    state: np.ndarray
    action_discrete: int
    action_continous: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_discrete: t.Optional[int] = None
    next_action_continous: t.Optional[float] = None


@dataclass
class Batch:
    """Defines the batch returned from memory for the platform environment"""

    states: torch.tensor
    actions_discrete: torch.tensor
    action_parameterized: torch.tensor
    rewards: torch.tensor
    next_states: torch.tensor
    done: torch.tensor
    next_actions_discrete: t.Optional[torch.tensor] = None
    next_action_parameterized: t.Optional[torch.tensor] = None

    def __len__(self) -> int:
        return len(self.actions_discrete)


class Memory:
    def __init__(self, max_size: int, batch_size: int) -> None:
        """Defines the memory of an agent, storing transitions in a queue structure.

        Args:
            max_size (int): The maximum number of transitions to store.
            batch_size (int): The batch size returned when get_batch is called.
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self._memory = deque()

    def __len__(self) -> int:
        return len(self._memory)

    def add(self, transition: Transition) -> None:
        """Appends a transition to the memory.

        Args:
            transition (Transition): A transition from the environment.
        """
        if len(self._memory) + 1 > self.max_size:
            self._memory.pop()
        self._memory.append(transition)

    def get_batch(self) -> Batch:
        """Returns a batch of stacked transitions.

        Returns:
            Batch: A batch of experiences.
        """
        if self.batch_size > len(self._memory):
            logger.warning(
                "The passed batch size exceeds the current memory length. Returning memory contents only."
            )
        batch = random.choices(self._memory, k=self.batch_size)
        state_batch = [torch.from_numpy(transition.state) for transition in batch]
        next_state_batch = [
            torch.from_numpy(transition.next_state) for transition in batch
        ]
        action_discrete_batch = torch.tensor(
            [transition.action_discrete for transition in batch]
        )
        action_continous_batch = [
            torch.from_numpy(transition.action_continous) for transition in batch
        ]
        rewards_batch = torch.tensor([transition.reward for transition in batch])
        done = torch.tensor([int(transition.done) for transition in batch])
        next_actions_discrete_batch = None
        next_action_parameterized_batch = None
        uses_next_action = all(
            [
                False if transition.next_action_continous is None else True
                for transition in batch
            ]
        )
        if uses_next_action:
            next_actions_discrete_batch = torch.tensor(
                [transition.next_action_discrete for transition in batch]
            )
            next_action_parameterized_batch = torch.tensor(
                [transition.next_action_continous for transition in batch]
            )

        return Batch(
            states=torch.vstack(state_batch).double(),
            next_states=torch.vstack(next_state_batch).double(),
            actions_discrete=action_discrete_batch,
            action_parameterized=torch.vstack(action_continous_batch).double(),
            rewards=rewards_batch.double(),
            done=done,
            next_action_parameterized=next_action_parameterized_batch,
            next_actions_discrete=next_actions_discrete_batch,
        )

    def reset(self) -> None:
        """Resets the memory."""
        self._memory = deque()
