from hydra.core.config_store import ConfigStore

from rl.conf.agents import MPQAgentConf
from rl.conf.entrypoints import InferenceEntrypointConf, TrainEntrypointConf
from rl.conf.mpq import MultiPassQConf, ParamActorConf

CONFIG_STORE = ConfigStore.instance()


def register_all():
    CONFIG_STORE.store(
        name=MPQAgentConf.__name__,
        package="rl.conf",
        provider="rl",
        group="agents",
        node=MPQAgentConf,
    )
    CONFIG_STORE.store(
        name=ParamActorConf.__name__,
        package="rl.conf",
        provider="rl",
        group="actors",
        node=ParamActorConf,
    )
    CONFIG_STORE.store(
        name=MultiPassQConf.__name__,
        package="rl.conf",
        provider="rl",
        group="q_networks",
        node=MultiPassQConf,
    )
    CONFIG_STORE.store(
        name=TrainEntrypointConf.__name__,
        package="rl.conf",
        provider="rl",
        group="entrypoints",
        node=TrainEntrypointConf,
    )
    CONFIG_STORE.store(
        name=InferenceEntrypointConf.__name__,
        package="rl.conf",
        provider="rl",
        group="entrypoints",
        node=InferenceEntrypointConf,
    )
