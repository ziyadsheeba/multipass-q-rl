import typing as t
from dataclasses import dataclass

from omegaconf import MISSING

from rl.conf.agents import AgentConf


@dataclass
class TrainEntrypointConf:
    agent: AgentConf = MISSING
    n_episodes: int = MISSING
    render_episodes_frequency: int = MISSING
    eval_episodes_frequency: int = MISSING
    use_wrapper: bool = True
    experiment_name: str = "Default Training Experiment"
    run_name: t.Optional[str] = None


@dataclass
class InferenceEntrypointConf:
    training_run_id: str = MISSING
    experiment_name: str = "Default Inference Experiment"
