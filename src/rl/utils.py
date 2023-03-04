import torch.nn as nn


def soft_update_target_network(
    source_network: nn.Module, target_network: nn.Module, tau: float
) -> None:
    """Performs polyak averaging on the target network with a factor tau.

    Args:
        source_network (nn.Module): The reference network.
        target_network (nn.Module): The network to be updated.
        tau (float): The convex combination factor.
    """
    for target_param, param in zip(
        target_network.parameters(), source_network.parameters()
    ):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def get_network_gradient_norm(model: nn.Module) -> float:
    """Returns the sum of norms of gradients of a model."""
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def get_network_gradient_max_norm(model: nn.Module) -> float:
    """Returns the maximum norm of gradients of a model."""
    norms = []
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        norms.append(param_norm.item())
    return max(norms)
