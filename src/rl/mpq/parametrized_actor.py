import typing as t

import torch
import torch.nn as nn


class ParamActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_parameter_sizes: t.List[int],
        hidden_layer_sizes: t.List[int],
    ) -> None:
        """Implements a parametrized action actor, as proposed in https://arxiv.org/pdf/1905.04388.pdf.

        Args:
            state_dim (int): State space dimension.
            action_parameter_sizes (t.List[int]): A list containing the dimensionality of each parameterized action.
                It will be assumed that the list can be indexed by the corresponding discrete action.
            hidden_layer_sizes (t.List[int]): A list containing the dimensionality of each hidden layer.

        Raises:
            ValueError: If the action_parameter_sizes don't have a length of action_dim.
            ValueError: If no hidden layers are passed (empty list).
        """
        super().__init__()
        if len(hidden_layer_sizes) < 1:
            raise ValueError("The hidden layers list must have a length of at least 1.")

        self.state_dim = state_dim
        self.action_parameter_dim = sum(action_parameter_sizes)
        self.base_network = self._build_network(hidden_layer_sizes=hidden_layer_sizes)

    def _build_network(
        self, hidden_layer_sizes: t.List[int]
    ) -> t.Tuple[nn.Sequential, nn.Linear]:
        """Returns the base_network used for the parameters actor as propossed in
            https://arxiv.org/pdf/1905.04388.pdf.

        Note, this implementation dis-regards the passthrough-layer proposed in the
        original paper.

        Args:
            hidden_layer_sizes (t.List[int]): A list of hidden sizes.

        Returns:
            t.Tuple[nn.Sequential, nn.Linear]: The first index contains a ReLU
            activated MLP taking states and returning parametrized actions.
            The second index contains a linear passthrough layer used
            to stabilize the network.
        """
        layers = [
            nn.Linear(self.state_dim, hidden_layer_sizes[0]),
            nn.ReLU(),
        ]
        nn.init.zeros_(layers[-2].bias)
        nn.init.kaiming_normal_(layers[-2].weight, nonlinearity="relu")
        for i in range(1, len(hidden_layer_sizes)):
            layer = nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i])
            layers.extend([layer, nn.ReLU()])
            nn.init.zeros_(layers[-2].bias)
            nn.init.kaiming_normal_(layers[-2].weight, nonlinearity="relu")
        final_layer = nn.Linear(hidden_layer_sizes[-1], self.action_parameter_dim)
        nn.init.zeros_(final_layer.bias)
        nn.init.normal_(final_layer.weight, std=0.0001)
        layers.extend([final_layer])
        return nn.Sequential(*layers)

    def forward(self, states: torch.tensor) -> torch.tensor:
        """Returns the predicted parametrized actions normalized in the range of [0, 1].

        Args:
            states (torch.tensor): A batch of states.

        Returns:
            torch.tensor: The parameterized actions in the range of (0,1), each.
        """
        x = self.base_network(states)
        return x
