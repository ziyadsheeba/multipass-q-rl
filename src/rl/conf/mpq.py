import typing as t
from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class MultiPassQConf:
    _target_: str = "rl.mpq.q_network.MultiPassQ"
    state_dim: int = MISSING
    action_dim: int = MISSING
    action_parameter_sizes: t.List[int] = MISSING
    hidden_layer_sizes: t.List[int] = field(default_factory=lambda: [100])


@dataclass
class ParamActorConf:
    _target_: str = "rl.mpq.parametrized_actor.ParamActor"
    state_dim: int = MISSING
    action_parameter_sizes: t.List[int] = MISSING
    hidden_layer_sizes: t.List[int] = field(default_factory=lambda: [100])
