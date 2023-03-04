import typing as t

import numpy as np
import torch
import torch.nn as nn


class MultiPassQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_parameter_sizes: t.List[int],
        hidden_layer_sizes: t.List[int],
    ) -> None:
        """Implements a Multipass Q-network proposed in https://arxiv.org/pdf/1905.04388.pdf.

        Args:
            state_dim (int): State space dimension.
            action_dim (int): Action space dimension.
            action_parameter_sizes (t.List[int]): A list containing the dimensionality of each parameterized action.
                It will be assumed that the list can be indexed by the corresponding discrete action.
            hidden_layer_sizes (t.List[int]): A list containing the dimensionality of each hidden layer.

        Raises:
            ValueError: If the action_parameter_sizes don't have the length of action_dim.
            ValueError: If no hidden layers are passed (empty list).
        """
        super().__init__()
        if len(action_parameter_sizes) != action_dim:
            raise ValueError(
                "The action parameter sizes list must match the dimension of the discrete action space"
            )
        if len(hidden_layer_sizes) < 1:
            raise ValueError("The hidden layers list must have a length of at least 1.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_parameter_sizes = action_parameter_sizes
        self.action_parameter_dim = sum(action_parameter_sizes)
        self.base_network = self._build_network(hidden_layer_sizes=hidden_layer_sizes)
        self.offsets = self._get_param_action_index_offsets()

    def _build_network(self, hidden_layer_sizes: t.List[int]) -> nn.Sequential:
        """Returns the base network used for the Q function.

        Args:
            hidden_layer_sizes (t.List[int]): A list of hidden sizes.

        Returns:
            nn.Sequential: A ReLU activated MLP, taking states and parametrized
                actions as input, and returning the Q-values for each action.
        """
        layers = [
            nn.Linear(
                self.state_dim + self.action_parameter_dim, hidden_layer_sizes[0]
            ),
            nn.ReLU(),
        ]
        nn.init.kaiming_normal_(layers[-2].weight, nonlinearity="relu")
        nn.init.zeros_(layers[-2].bias)
        for i in range(1, len(hidden_layer_sizes)):
            layers.extend(
                [nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]), nn.ReLU()]
            )
            nn.init.kaiming_normal_(layers[-2].weight, nonlinearity="relu")
            nn.init.zeros_(layers[-2].bias)
        layers.append(nn.Linear(hidden_layer_sizes[-1], self.action_dim))
        nn.init.normal_(layers[-1].weight, mean=0.0, std=0.0001)
        nn.init.zeros_(layers[-1].bias)
        return nn.Sequential(*layers)

    def _get_param_action_index_offsets(self) -> t.List[int]:
        """Returns offsets that help index the action parameters
        depending on the size of each parameter.

        Returns:
            t.List[int]: A list of offsets such that action_param[offset[i]:offset[i+1]]
            returns the parameterized action for the discrete action i.
        """
        offsets = np.cumsum(self.action_parameter_sizes).tolist()
        return [0] + offsets

    def forward(
        self, states: torch.tensor, action_parameters: torch.tensor
    ) -> torch.tensor:
        """Returns the Q-value of each discrete action, at a given state and action parameter.

        Args:
            states (torch.tensor): A batch of states.
            action_parameters (torch.tensor): The action parameters from the
                actor base_network.

        Returns:
            torch.tensor: the Q-value of each discrete action, at a given state and action parameter.
        """
        batch_size = states.shape[0]
        x = torch.cat([states, torch.zeros_like(action_parameters)], axis=1)
        x = x.repeat(self.action_dim, 1)
        for a in range(self.action_dim):
            x[
                a * batch_size : (a + 1) * batch_size,
                self.state_dim + self.offsets[a] : self.state_dim + self.offsets[a + 1],
            ] = action_parameters[:, self.offsets[a] : self.offsets[a + 1]]
        q_all = self.base_network(x)
        q = []
        for a in range(self.action_dim):
            q_a = q_all[a * batch_size : (a + 1) * batch_size, a]
            if len(q_a.shape) == 1:
                q_a = q_a.unsqueeze(1)
            q.append(q_a)
        q = torch.cat(q, dim=1)
        return q
