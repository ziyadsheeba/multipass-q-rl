import torch
from rl.mpq.q_network import MultiPassQ

def test_q_network_return_shape() -> None:
    """Tests that the multipass q value network returns the correct shape.
    """
    state_dim = 9
    action_dim = 3
    action_parameter_sizes = [1,1,1]
    hidden_layer_sizes = [100, 100]
    batch_size = 100
    q_net = MultiPassQ(
        state_dim=state_dim,
        action_dim=action_dim,
        action_parameter_sizes=action_parameter_sizes,
        hidden_layer_sizes=hidden_layer_sizes
    )
    states = torch.rand(size=(batch_size, state_dim))
    action_parameters = torch.cat([torch.rand(size = (batch_size, size)) for size in action_parameter_sizes], axis = 1)
    q_vals = q_net(states, action_parameters)
    assert q_vals.shape == (batch_size, action_dim)