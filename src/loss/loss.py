from typing import Tuple, Callable

import torch


# adversarial losses
def hinge_loss() -> Tuple[Callable, Callable]:
    """Hinge GAN loss https://arxiv.org/abs/1705.02894v2

    Returns:
        Tuple[Callable, Callable]: discriminator loss, generator loss
    """

    def d_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.relu(1 - real_pred).mean()
            + torch.nn.functional.relu(1 + fake_pred).mean()
        )

    def g_loss(fake_pred: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(1 - fake_pred).mean()

    return d_loss, g_loss


def non_saturating_loss() -> Tuple[Callable, Callable]:
    """Non saturating GAN loss

    Returns:
        Tuple[Callable, Callable]: discriminator loss, generator loss
    """

    def d_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.softplus(-real_pred).mean()
            + torch.nn.functional.softplus(fake_pred).mean()
        )

    def g_loss(fake_pred: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(-fake_pred).mean()

    return d_loss, g_loss


def lsgan_loss() -> Tuple[Callable, Callable]:
    """LSGAN (least square) loss

    Returns:
        Tuple[Callable, Callable]: discriminator loss, generator loss
    """

    def d_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        return (
            torch.square((1 - real_pred)**2).mean()
            + torch.square(fake_pred**2).mean()
        )

    def g_loss(fake_pred: torch.Tensor) -> torch.Tensor:
        return torch.square((1 - fake_pred)**2).mean()

    return d_loss, g_loss


def get_adversarial_losses(type: str = "hinge"):

    ADV_LOSSES = {
        "hinge": hinge_loss,
        "non_saturating": non_saturating_loss,
        "lsgan": lsgan_loss,
    }

    assert type.lower() in ADV_LOSSES, "Adversarial loss {type} is not implemented"
    return ADV_LOSSES[type]()


# regularizers
def r1_loss(output, input):
    grad, = torch.autograd.grad(
        outputs=output.sum(), inputs=input, create_graph=True
    )
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def get_regularizer(type: str = "r1"):

    REGULARIZERS = {
        "r1": r1_loss,
    }

    assert type.lower() in REGULARIZERS, "Regularizer {type} is not implemented"
    return REGULARIZERS[type]
