import gym
import gym_platform
from rl.platform_utils import create_transition
from rl.memory import Memory
from rl.wrappers import PlatformWrapper
import torch
import numpy as np

def test_transition_creation() -> None:
    """Tests that the transitions are created as expected.
    """
    env = PlatformWrapper(gym.make("Platform-v0"))
    s = env.reset()
    action = (np.random.choice(3), np.random.uniform(low=-1, high=1, size=(3,)))
    s_prime, reward, done, _ = env.step(action=action)
    transition = create_transition(
        state=s,
        next_state=s_prime,
        action=action,
        reward=reward,
        done=done
    )
    assert transition.action_continous[[action[0]]] == action[1][action[0]].item()
    assert transition.action_discrete == action[0]
    assert np.allclose(transition.state, s)
    assert np.allclose(transition.next_state, s_prime)

def test_memory_batch() -> None:
    """Tests that the batches are created from memory as expected.
    """
    batch_size = 2
    max_size = 2
    memory = Memory(batch_size=batch_size, max_size=max_size)
    env = PlatformWrapper(gym.make("Platform-v0"))
    s = env.reset()
    action = (np.random.choice(3), np.random.uniform(low=-1, high=1, size=(3,)))
    s_prime, reward, done, _ = env.step(action=action)
    transition = create_transition(
        state=s,
        next_state=s_prime,
        action=action,
        reward=reward,
        done=done
    )
    memory.add(transition=transition)
    batch = memory.get_batch()
    
    assert batch.states.shape[0] == batch_size
    assert batch.next_states.shape[0] == batch_size
    assert batch.actions_discrete.shape == (batch_size,)
    assert batch.action_parameterized.shape == (batch_size,3)

def test_memory_buffer_size() -> None:
    """Tests that the memory size works as expected.
    """
    batch_size = 2
    max_size = 1
    memory = Memory(batch_size=batch_size, max_size=max_size)
    env = PlatformWrapper(gym.make("Platform-v0"))
    
    for i in range(2):
        s = env.reset()
        action = (np.random.choice(3), np.random.uniform(low=-1, high=1, size=(3,)))
        s_prime, reward, done, _ = env.step(action=action)
        transition = create_transition(
            state=s,
            next_state=s_prime,
            action=action,
            reward=reward,
            done=done,
        )
        memory.add(transition=transition)
    batch = memory.get_batch()
    assert torch.allclose(batch.states[0,:], batch.states[1, :])
    assert torch.allclose(batch.next_states[0,:], batch.next_states[1,:])
    assert batch.actions_discrete[0] == batch.actions_discrete[1]
    assert torch.allclose(batch.action_parameterized, batch.action_parameterized)