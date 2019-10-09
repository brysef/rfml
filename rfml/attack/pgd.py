"""Functional interface to a PGD attack.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.functional import cross_entropy

from typing import Union

# Internal Includes
from rfml.nn.model import Model
from rfml.ptradio import Slicer

from .utils import _convert_or_throw, _infer_input_size, _normalize
from .utils import _random_uniform_start, _compute_multiplier


def pgd(
    x: torch.Tensor,
    y: Union[torch.Tensor, int],
    net: Model,
    spr: float,
    k: int,
    input_size: int = None,
    sps: int = 8,
) -> torch.Tensor:
    """Projected Gradient Descent attack

    Args:
        x (torch.Tensor): Continuous input signal (BxCxIQxN)
        y (Union[torch.Tensor, int]): The categorical (integer) label for the input
                                      signals.  This can either be a single integer,
                                      which is then assumed to be the label for all
                                      inputs, or it can be a a Tensor (B) specifying a
                                      label for each batch of x.
        k (int): Number of iterations to use for the attack.
        net (Model): Classification model to use for computing the gradient signal.
        input_size (int, optional): Number of time samples that net takes in at a time.
                                    If not provided, it is inferred from the x shape.
                                    Defaults to None.
        spr (float): Signal-to-Perturbation ratio (SPR) in dB that is used to scale the
                     power of the perturbation signal crafted and applied.
        sps (int, optional): Samples per symbol (sps) needed to compute the correct
                             scaling for achieving the desired spr. Defaults to 8.

    Returns:
        torch.Tensor: Perturbed signal (x + p) which is formatted as BxCxIQxN

    .. warn::

        This function assumes that Es is 1 when scaling the perturbation to achieve a
        desired SPR.  Therefore, it will first rescale the underlying example to ensure
        that is true.  Generally, this will not cause an issue because the model that
        uses the example will rescale the signal anyways based on its own normalization.

    Reference:
        Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu,
        "Towards Deep Learning Models Resistant to Adversarial Attacks",
        https://arxiv.org/abs/1706.06083
    """
    x, y = _convert_or_throw(x=x, y=y)
    if k < 1:
        raise ValueError("K must be a positive integer -- you gave {}".format(k))
    input_size = _infer_input_size(x=x, input_size=input_size)

    x = _normalize(x=x, sps=sps)
    x = Slicer(width=input_size)(x)
    adv_x = _random_uniform_start(x=x, spr=spr, sps=sps)

    # Create step_size based on the overall distance we want to travel
    eps = _compute_multiplier(spr=spr, sps=sps)
    step_size = eps / float(k)

    # Compute the bounds on the feasible solution (the ball around the input)
    lower_bound = x - eps
    upper_bound = x + eps

    # Put the inputs/outputs onto the most probable device that the model is currently
    # on -- this could fail if the model gets split amongst multiple devies, but, that
    # doesn't happen in this code.
    adv_x = adv_x.to(net.device)
    y = y.to(net.device)
    upper_bound = upper_bound.to(net.device)
    lower_bound = lower_bound.to(net.device)

    # Ensure the model is in eval mode so that batch norm/dropout etc. doesn't take
    # effect -- in order to be transparent to the caller we need to restore the state
    # at the end.
    set_training = net.training
    if set_training:
        net.eval()

    for _k in range(k):
        # Perform forward/backward pass to get the gradient at the input
        adv_x.requires_grad = True
        _y = net(adv_x)
        loss = cross_entropy(_y, y)
        loss.backward()

        # Take the sign of the gradient that can be scaled later
        sg = torch.sign(adv_x.grad.data)

        # Take a step in the direction of the signed gradient then project back onto the
        # feasible set of solutions (the ball around the original example) by clipping
        adv_x = adv_x + step_size * sg
        # torch.clamp only supports a single value -- therefore we'd have to extract the
        # perturbation first, by subtracting the original example, then clip the
        # perturbation, then recreate the adversarial example by adding the perturbation
        # to the natural example -- this methodology represents one fewer function call
        adv_x = torch.max(adv_x, lower_bound)
        adv_x = torch.min(adv_x, upper_bound)
        # Ensure that gradients are still tracked on this new adversarial example
        adv_x = Variable(adv_x)

    # Restore the network state so the caller never notices the change
    if set_training:
        net.train()

    return adv_x
