import typing as t
from dataclasses import dataclass

from omegaconf import MISSING

from rl.conf.mpq import MultiPassQConf, ParamActorConf


@dataclass
class AgentConf:
    _target_: str = MISSING
    state_dim: int = MISSING
    action_dim: int = MISSING
    gamma: float = MISSING
    memory_size: int = MISSING
    memory_batch_size: int = MISSING
    seed: int = MISSING


@dataclass
class MPQAgentConf(AgentConf):
    _target_: str = "rl.agents.MPQAgent"
    q_network: MultiPassQConf = MISSING
    param_actor: ParamActorConf = MISSING
    tau_q: float = 0.01
    tau_param_actor: float = 0.001
    learning_rate_q: float = 1e-4
    learning_rate_param_actor: float = 1e-5
    epsilon_min: float = MISSING
    epsilon_max: float = MISSING
    decay_epsilon_episodes: int = MISSING
    use_cuda: bool = True
    update_steps_burn_in: int = MISSING
    update_frequency_steps: int = MISSING
