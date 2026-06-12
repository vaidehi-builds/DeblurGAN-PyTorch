import torch
from torch import autograd


def gradient_penalty(
    critic,
    real,
    fake,
    device
):
    batch_size = real.size(0)

    epsilon = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=device
    )

    interpolated = (
        epsilon * real +
        (1 - epsilon) * fake
    )

    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    gradients = autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(
        gradients.size(0),
        -1
    )

    gradient_norm = gradients.norm(
        2,
        dim=1
    )

    gp = (
        (gradient_norm - 1) ** 2
    ).mean()
    
    return gp
def critic_loss(
    critic,
    real,
    fake,
    lambda_gp,
    device
):
    real_scores = critic(real)

    fake_scores = critic(
        fake.detach()
    )

    gp = gradient_penalty(
        critic,
        real,
        fake.detach(),
        device
    )

    loss = (
        fake_scores.mean()
        - real_scores.mean()
        + lambda_gp * gp
    )

    return loss


def generator_loss(
    critic,
    fake
):
    fake_scores = critic(fake)

    loss = -fake_scores.mean()

    return loss