import torch
from rl.mpq.parametrized_actor import ParamActor

def test_param_actor_return_shape() -> None:
    """Tests that the parameters actor network returns the correct shape.
    """
    state_dim = 9
    action_parameter_sizes = [1,1,1]
    hidden_layer_sizes = [100, 100]
    batch_size = 100
    actor = ParamActor(
        state_dim=state_dim,
        action_parameter_sizes=action_parameter_sizes,
        hidden_layer_sizes=hidden_layer_sizes
    )
    states = torch.rand(size=(batch_size, state_dim))
    param_actions = actor(states)
    assert param_actions.shape == (batch_size, sum(action_parameter_sizes))