import logging
import typing as t
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from rl.memory import Memory, Transition
from rl.mpq.parametrized_actor import ParamActor
from rl.mpq.q_network import MultiPassQ
from rl.utils import soft_update_target_network

State = t.TypeVar("State")
Action = t.TypeVar("Action")

logger = logging.getLogger(__name__)


class Agent(ABC):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float,
        memory_size: int,
        memory_batch_size: int,
        seed: int,
    ) -> None:
        """Instantiates an agent.

        Args:
            state_dim (int): The state space dimensions.
            action_dim (int): The action space dimensions.
            gamma (float): The MDP's discounting factor.
            memory_size (int): The maximum memory size.
            memory_batch_size (int): The batch size sampled from memory.
            seed (int): The random seed.
        """
        self.memory = Memory(max_size=memory_size, batch_size=memory_batch_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.np_random = np.random.default_rng(seed)

    def store(self, transition: Transition) -> None:
        """Stores a transition in the agent's memory."""
        self.memory.add(transition=transition)

    @abstractmethod
    def update(self) -> None:
        """Updates the agent's policy/value networks."""
        pass

    @abstractmethod
    def get_action(self, state: State) -> Action:
        """Returns the action under the current policy for a given state.

        Args:
            state (State): The current state.

        Returns:
            Action: The action to play under the current policy.
        """
        pass


class MPQAgent(Agent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float,
        memory_size: int,
        memory_batch_size: int,
        seed: int,
        q_network: MultiPassQ,
        param_actor: ParamActor,
        tau_q: float,
        tau_param_actor: float,
        learning_rate_q: float,
        learning_rate_param_actor: float,
        epsilon_min: float,
        epsilon_max: float,
        decay_epsilon_episodes: int,
        update_steps_burn_in: int,
        update_frequency_steps: int,
        use_cuda: bool,
    ) -> None:
        """Instantiates an multipass parameterized-q agent as proposed in
        https://arxiv.org/pdf/1905.04388.pdf.


        Args:
            state_dim (int): The observation space dimension.
            action_dim (int): The discrete action space dimension.
            gamma (float): The MDP's discounting factor.
            memory_size (int): The maximum number of transitions to store in memory.
            memory_batch_size (int): The batch size retrieved from memory, used for agent training.
            seed (int): The random seed.
            q_network (MultiPassQ): A multipass q network.
            param_actor (ParamActor): The parametrized actor network.
            tau_q (float): The polyak averaging factor for updating the target q networks.
            tau_param_actor (float): The polyak averaging factor for updating the actor network.
            learning_rate_q (float): The learning rate for the q network.
            learning_rate_param_actor (float): The learning rate for the parametrized actor.
            epsilon_min (float): The minimum exploration noise during training. Must be greater than 0.
            epsilon_max (float): The maximum exploration noise during training. Must be less than 1.
            decay_epsilon_episodes (int): The number of episodes to decay epsilon from maximum to minimum.
            update_steps_burn_in (int): The number of steps to wait before updating the agent.
            update_frequency_steps (int): The frequency in steps (transitions) to update the agent.
            use_cuda (bool): Whether to use cuda, if available.
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            memory_size=memory_size,
            memory_batch_size=memory_batch_size,
            seed=seed,
        )
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.q_network = q_network.to(self.device).double()
        self.q_network_target = deepcopy(q_network).to(self.device).double()
        self.q_network_target.eval()
        self.param_actor = param_actor.to(self.device).double()
        self.param_actor_target = deepcopy(param_actor).to(self.device).double()
        self.param_actor_target.eval()
        self.tau_q = tau_q
        self.tau_param_actor = tau_param_actor
        self.learning_rate_q = learning_rate_q
        self.learning_rate_param_actor = learning_rate_param_actor
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.decay_epsilon_episodes = decay_epsilon_episodes

        self.update_steps_burn_in = update_steps_burn_in
        self.update_frequency_steps = update_frequency_steps
        self.loss_func = nn.MSELoss()

        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate_q
        )
        self.actor_param_optimizer = torch.optim.Adam(
            self.param_actor.parameters(), lr=self.learning_rate_param_actor
        )
        self._param_action_max = torch.ones(
            (self.param_actor.action_parameter_dim,)
        ).double()
        self._param_action_min = -torch.ones(
            (self.param_actor.action_parameter_dim,)
        ).double()
        self._param_action_rng = self._param_action_max - self._param_action_min

        self._steps = 0
        self._n_episode = 0
        self.eval_mode = False

    def step(self, transition: Transition):
        """Stores a transition in memory, and updates the agent
        if a condition on the number of steps taken is met.

        Args:
            transition (Transition): The transition to store in memory.
        """
        if not self.eval_mode:
            self.store(transition)
            self._steps += 1
            if (
                self._steps > self.update_steps_burn_in
                and self._steps > self.memory.batch_size
                and self._steps % self.update_frequency_steps == 0
            ):
                self.update()

    def eval(self) -> None:
        """Sets the agent in evaluation mode."""
        self.param_actor.eval()
        self.q_network.eval()
        self.eval_mode = True

    def train(self) -> None:
        """Sets the agent in training mode."""
        self.param_actor.train()
        self.q_network.train()
        self.eval_mode = False

    def increment_episode(self) -> None:
        """Increments an episode counter, and decays the exploration noise linearly between min and max."""
        self._n_episode += 1
        if self._n_episode < self.decay_epsilon_episodes:
            self.epsilon = self.epsilon_max - (self.epsilon_max - self.epsilon_min) * (
                self._n_episode / self.decay_epsilon_episodes
            )
        else:
            self.epsilon = self.epsilon_min

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> t.Tuple[int, np.ndarray]:
        """Retrieves the agent's proposed action given a state.

        Note, if you want to retrieve the action without any exploration noise,
        make sure to set the agent in eval mode by calling agent.eval(). To
        return back

        Args:
            state (np.ndarray): The agent's current state.

        Returns:
            t.Tuple[int, np.ndarray]: The proposed action with exploration
            noise if agent.train() is activated, else the greedy action
            wrt to the Q-values is returned.
        """
        state = torch.from_numpy(state).double()
        action_parameters = self.param_actor.forward(state)
        rnd = self.np_random.uniform()
        if rnd < self.epsilon and not self.eval_mode:
            action = self.np_random.choice(np.arange(self.action_dim))
            action_parameters = (
                2 * np.random.uniform(size=(self.param_actor.action_parameter_dim,)) - 1
            )
        else:
            q = self.q_network.forward(
                state.unsqueeze(0), action_parameters.unsqueeze(0)
            )
            action = torch.argmax(q).item()
            action_parameters = action_parameters.detach().cpu().numpy().squeeze()
        return action, action_parameters

    @torch.no_grad()
    def _invert_gradients(
        self, gradient: torch.tensor, values: torch.tensor
    ) -> torch.tensor:
        """Applies gradient inversion as proposed in https://arxiv.org/abs/1511.04143
        to ensure that the values are kept in their bounded range.

        Args:
            gradient (torch.tensor): The gradient tensor.
            values (torch.tensor): The bounded values.

        Returns:
            torch.tensor: A tensor of inverted gradients.
        """
        values = values.cpu()
        gradient = gradient.cpu()
        index = gradient > 0
        gradient[index] *= (
            index.float() * (self._param_action_max - values) / self._param_action_rng
        )[index]
        gradient[~index] *= (
            (~index).float()
            * (values - self._param_action_min)
            / self._param_action_rng
        )[~index]
        return gradient

    def update(self) -> None:
        """Updates the agent's networks using a batch of transitions."""
        batch = self.memory.get_batch()
        states, next_states, done, actions_discrete, action_parameterized, rewards = (
            batch.states.to(self.device),
            batch.next_states.to(self.device),
            batch.done.to(self.device),
            batch.actions_discrete.to(self.device),
            batch.action_parameterized.to(self.device),
            batch.rewards.to(self.device),
        )

        # q base_network loss
        with torch.no_grad():
            pred_next_action_parameters = self.param_actor_target.forward(next_states)
            q_prime = self.q_network_target(next_states, pred_next_action_parameters)
            q_prime_max = torch.max(q_prime, 1, keepdim=True)[0].squeeze()
            target = rewards + (1 - done) * self.gamma * q_prime_max

        qvals = self.q_network(states, action_parameterized)
        loss_q = self.loss_func(
            qvals.gather(1, actions_discrete.view(-1, 1)).squeeze(),
            target,
        )
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # parameters actor loss
        with torch.no_grad():
            param_actions = self.param_actor(states)

        param_actions.requires_grad = True
        qvals = self.q_network(states, param_actions)
        loss_actor = torch.mean(torch.sum(qvals, 1))
        self.q_network.zero_grad()
        loss_actor.backward()

        delta_a = deepcopy(param_actions.grad.data)  # derivative of q wrt param actions
        param_actions = self.param_actor(states)

        delta_a[:] = self._invert_gradients(
            delta_a,
            param_actions,
        )
        out = -torch.mul(delta_a, param_actions)
        self.param_actor.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        self.actor_param_optimizer.step()

        soft_update_target_network(self.q_network, self.q_network_target, self.tau_q)
        soft_update_target_network(
            self.param_actor, self.param_actor_target, self.tau_param_actor
        )

    def act_and_explain(
        self, state: np.ndarray
    ) -> t.Tuple[int, np.ndarray, np.ndarray]:
        """Acts and explains the action using feature importances of the observation space.

        The explaination is done by backpropagating the Q-value corresponding to the played action
        back to the states, and then normalizing the absolute value of the gradients between [0,1].
        (GradCAM)

        Args:
            state (np.ndarray): The agent's current state.

        Returns:
            t.Tuple[int, np.ndarray, np.ndarray]: The discrete action, the parametrized action and
            the feature importance.
        """
        if not self.eval_mode:
            logger.warning(
                "act_and_explain act called while the agent is not in eval mode. Setting agent to eval mode"
            )
            self.eval()
        state = torch.from_numpy(state).double()
        state.requires_grad = True
        action_parameters = self.param_actor.forward(state)
        q = self.q_network.forward(state.unsqueeze(0), action_parameters.unsqueeze(0))
        action = torch.argmax(q).item()
        optimal_q = q.squeeze()[action]
        self.q_network.zero_grad()
        optimal_q.backward()
        gradients_abs = torch.abs(state.grad.data)
        feature_importance = (gradients_abs / torch.max(gradients_abs)).numpy()
        action_parameters = action_parameters.detach().cpu().numpy().squeeze()
        return action, action_parameters, feature_importance

    def save_models(self, dir: str) -> None:
        """Saves the agent's models under a given directory.

        Args:
            dir (str): The directory to save the models to.
        """
        torch.save(self.q_network.state_dict(), f"{dir}/q_network.pt")
        torch.save(self.param_actor.state_dict(), f"{dir}/param_actor.pt")
        torch.save(self.q_network_target.state_dict(), f"{dir}/q_network_target.pt")
        torch.save(self.param_actor_target.state_dict(), f"{dir}/param_actor_target.pt")

    def load_models(self, dir: str) -> None:
        """Loads the agent's models from a given directory.s

        Args:
            dir (str): The directory to retreive the models from.
        """
        self.q_network.load_state_dict(torch.load(f"{dir}/q_network.pt"))
        self.param_actor.load_state_dict(torch.load(f"{dir}/param_actor.pt"))
        self.q_network_target.load_state_dict(torch.load(f"{dir}/q_network_target.pt"))
        self.param_actor_target.load_state_dict(
            torch.load(f"{dir}/param_actor_target.pt")
        )
