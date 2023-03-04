import logging
import tempfile
import typing as t
import uuid
from abc import ABC, abstractmethod

import gym
import gym_platform
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from rl.conf.entrypoints import InferenceEntrypointConf, TrainEntrypointConf
from rl.constants import PLATFORM_FEATURE_MAPPINGS
from rl.platform_utils import create_transition
from rl.wrappers import PlatformWrapper

logger = logging.getLogger(__name__)
plt.style.use("ggplot")


class Entrypoint(ABC):
    @abstractmethod
    def _on_run_start(self) -> None:
        """Any callbacks that should be called prior to running the main piece of code."""
        pass

    @abstractmethod
    def _on_run_end(self) -> None:
        """Any callbacks that should be called after running the main piece of code."""
        pass

    @abstractmethod
    def _run(self) -> None:
        """Where the main code and logic should be put."""
        pass

    def __call__(self) -> None:
        self._on_run_start()
        self._run()
        self._on_run_end()


class TrainingEntrypoint(Entrypoint):
    """Wraps the main training code of an agent.

    This entrypoint does the following:
        1) Starts an mlflow run
        2) Instantiates an agent using the passed configs
        3) Runs a training loop
        4) Evaluates the agent every n steps and logs the best
        actor and critic networks.

    All artifacts are logged to mlflow.
    """

    def __init__(self, cfg: TrainEntrypointConf) -> None:
        """Instantiates a training entrypoint.

        Args:
            cfg (TrainEntrypointConf): The training entrypoint configuration.

        Raises:
            NotImplementedError: If the unwrapped environment was to be requested.
        """
        self.cfg = cfg
        if not cfg.use_wrapper:
            raise NotImplementedError("The unwrapped environment is not supported")
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.env = PlatformWrapper(gym.make("Platform-v0"))
        self.run: t.Optional[mlflow.ActiveRun] = None

    def _on_run_start(self) -> None:
        """Starts an mlflow experiments and logs the configs to the console."""
        logger.info("Training entrypoint started.")
        logger.debug(self.cfg)
        mlflow.set_experiment(self.cfg.experiment_name)
        self.run = mlflow.start_run(
            run_name=self.cfg.run_name,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = f"{tmp_dir}/config.yaml"
            with open(path, "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))
            mlflow.log_artifact(path)
        mlflow.log_params(self.cfg.agent)

    def _on_run_end(self) -> str:
        """Ends the mlflow run and logs info to the console."""
        run_id = self.run.info.run_id
        logger.info("Training entrypoint ended.")
        logger.info(f"Mlflow run id: {run_id}")
        mlflow.end_run()
        return run_id

    def rollout_episode(self, episode: int) -> float:
        """Rolls out an episode, collects experiences and
        and stores it in the agent's memory. The agent is
        updated in the agent.step() method.

        Note that when the agent is in eval mode, transitions are not
        stored in the agent's memory and no training will take place.

        Args:
            episode (int): The episode counter.

        Returns:
            float: The accumilated reward throughout the episode.
        """
        s = self.env.reset()
        done = False
        rewards = []
        while not done:
            action = self.agent.get_action(s)
            s_prime, reward, done, _ = self.env.step(action=action)
            if (
                episode % self.cfg.render_episodes_frequency == 0
                and self.agent.eval_mode
            ):
                self.env.render("rgb")
            transition = create_transition(
                state=s, next_state=s_prime, reward=reward, action=action, done=done
            )
            self.agent.step(transition)
            s = s_prime
            rewards.append(reward)
        rewards = sum(rewards)
        return rewards

    def log_models(self) -> None:
        """Logs the agent's models to mlflow"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.agent.save_models(dir=tmp_dir)
            mlflow.log_artifacts(tmp_dir)

    def _run(self) -> None:
        """Rolls out episodes, trains the agent and evaluates the agent at a given
        episode frequency.
        """
        cfg: TrainEntrypointConf = self.cfg
        t = trange(cfg.n_episodes, miniters=100, leave=True)
        reward_best = float("-inf")
        for episode in t:
            train_rewards = self.rollout_episode(episode=episode)
            mlflow.log_metric("train_reward", train_rewards, step=episode)
            if episode % cfg.eval_episodes_frequency == 0:
                self.agent.eval()
                eval_rewards = self.rollout_episode(episode=episode)
                mlflow.log_metric("eval_reward", eval_rewards, step=episode)
                self.agent.train()
                if eval_rewards > reward_best:
                    self.log_models()
                    reward_best = eval_rewards
                t.set_description(f"Total Reward: {eval_rewards}")
                t.refresh()
            self.agent.increment_episode()

    def __call__(self) -> str:
        self._on_run_start()
        self._run()
        run_id = self._on_run_end()
        return run_id


class InferenceEntrypoint(Entrypoint):
    """Wraps the main code used for inference with the agent.

    This entrypoint does the following:
        1) Loads the trained agent given an mlflow run id
        2) Runs the agent in the environment and renders the dynamics.
        3) Logs the state feature importance corresponding to the played
        action at everystep.
    """

    def __init__(self, cfg: InferenceEntrypointConf) -> None:
        """Instantiates an inference entrypoint.

        Args:
            cfg (InferenceEntrypointConf): The inference config.
        """
        self.cfg = cfg
        self.agent = self._load_agent()
        self.env = PlatformWrapper(gym.make("Platform-v0"))
        self.run: t.Optional[mlflow.ActiveRun] = None

    def _load_agent(self) -> DictConfig:
        """Loads the training config that was logged during
        the training run.

        Returns:
            DictConfig: The trainer config.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            mlflow.artifacts.download_artifacts(
                run_id=self.cfg.training_run_id, dst_path=tmp_dir
            )
            trainer_cfg = OmegaConf.load(f"{tmp_dir}/config.yaml")
            agent = hydra.utils.instantiate(trainer_cfg.agent)
            agent.load_models(tmp_dir)
        agent.eval()
        return agent

    def _on_run_start(self) -> None:
        """Starts an mlflow experiments and logs the configs."""
        logger.info("Inference entrypoint started.")
        logger.debug(self.cfg)
        mlflow.set_experiment(self.cfg.experiment_name)
        self.run = mlflow.start_run()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = f"{tmp_dir}/config.yaml"
            with open(path, "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))
            mlflow.log_artifact(path)
        mlflow.log_params(self.cfg)

    def _log_feature_importance(
        self, feature_importance: np.ndarray, step: int, env_snapshot: np.ndarray
    ) -> None:
        """Logs the feature importance that lead to the choice of an action to mlflow.

        Args:
            feature_importance (np.ndarray): A numpy array, with the size of the state dimension.
            step (int): The environment step.
            env_snapshot (np.ndarray): A snapshot of the environment prior to playing the action.
        """
        plt.figure(figsize=(15, 5))
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        axs[0].bar(list(PLATFORM_FEATURE_MAPPINGS.values()), feature_importance)
        axs[0].xaxis.set_tick_params(rotation=15)
        axs[0].set_title(f"Feature importance @step {step}")
        axs[0].set_xlabel("Feature Name")
        axs[0].set_ylabel("Feature Importance")
        axs[1].imshow(np.fliplr(env_snapshot))
        mlflow.log_figure(fig, f"feature_importance_{step}.png")
        plt.close()

    def _on_run_end(self) -> None:
        """Stops the mlflow experiment and prints the run id where the artifacts are logged."""
        run_id = self.run.info.run_id
        logger.info("Inference entrypoint ended.")
        logger.info(f"Mlflow run id: {run_id}")
        mlflow.end_run()

    def rollout_and_explain_episode(self) -> float:
        """Rolls out the agent in the environment for one episode and explains the agent's actions.

        Returns:
            float: The accumilated reward during the episode.
        """
        s = self.env.reset()
        done = False
        rewards = []
        step = 0
        while not done:
            action, action_param, feature_importance = self.agent.act_and_explain(s)
            self._log_feature_importance(
                feature_importance, step, self.env.render("rgb_array")
            )
            s_prime, reward, done, _ = self.env.step(action=(action, action_param))
            self.env.render("rgb")
            s = s_prime
            rewards.append(reward)
            step += 1
        rewards = sum(rewards)
        return rewards

    def _run(self) -> None:
        """Rolls out a rendered episode using a trained agent."""
        self.rollout_and_explain_episode()
