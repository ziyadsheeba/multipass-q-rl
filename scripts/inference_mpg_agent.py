import hydra

from rl.conf.entrypoints import InferenceEntrypointConf
from rl.constants import CONFIGS_PATH
from rl.entrypoints.entrypoints import InferenceEntrypoint


@hydra.main(
    version_base=None, config_path=CONFIGS_PATH, config_name="inference_entrypoint"
)
def main(cfg: InferenceEntrypointConf) -> None:
    inference_ep = InferenceEntrypoint(cfg)
    inference_ep()


if __name__ == "__main__":
    main()
