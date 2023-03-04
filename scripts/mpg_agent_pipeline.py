import hydra
from omegaconf import DictConfig

from rl.conf.entrypoints import InferenceEntrypointConf, TrainEntrypointConf
from rl.constants import CONFIGS_PATH
from rl.entrypoints.entrypoints import InferenceEntrypoint, TrainingEntrypoint


@hydra.main(
    version_base=None, config_path=CONFIGS_PATH, config_name="training_entrypoint"
)
def main(cfg: TrainEntrypointConf) -> None:
    trainer_ep = TrainingEntrypoint(cfg)
    run_id = trainer_ep()
    inference_ep = InferenceEntrypoint(
        DictConfig(InferenceEntrypointConf(training_run_id=run_id))
    )
    inference_ep()


if __name__ == "__main__":
    main()
