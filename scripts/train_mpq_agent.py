import hydra

from rl.conf.entrypoints import TrainEntrypointConf
from rl.constants import CONFIGS_PATH
from rl.entrypoints.entrypoints import TrainingEntrypoint


@hydra.main(
    version_base=None, config_path=CONFIGS_PATH, config_name="training_entrypoint"
)
def main(cfg: TrainEntrypointConf) -> None:
    trainer_ep = TrainingEntrypoint(cfg)
    trainer_ep()


if __name__ == "__main__":
    main()
