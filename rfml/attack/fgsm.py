"""Functional interface to an FGSM attack.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np

import torch
from torch.nn.functional import cross_entropy

from typing import Tuple, Union

# Internal Includes
import rfml.nn.F as F
from rfml.nn.model import Model
from rfml.ptradio import Slicer

from .utils import _convert_or_throw, _infer_input_size, _dither, _normalize
from .utils import _compute_multiplier


def fgsm(
    x: torch.Tensor,
    y: Union[torch.Tensor, int],
    net: Model,
    spr: float,
    input_size: int = None,
    sps: int = 8,
) -> torch.Tensor:
    """Create a perturbation using the Fast Gradient Sign Method (untargeted).

    This method performs an untargeted attack by:
        - Slicing the signal, x, into discrete examples of length input_size
        - Passing all examples into the neural network (net)
        - Computing the loss function (categorical cross entropy) with respect to the
            true class (y).
        - Backpropagating this back to the input to receive the gradient with respect
            to the input where the sign of the gradient can then be computed.
        - SPR is then used to scale the signed gradient to achieve a desired power.

    .. math::

            \\text{grad} = \\text{sign}(\\nabla_X \\mathcalP{L}(f(\\theta, X), y_s))

            P = \\sqrt{\\frac{10^{\\frac{-\\text{spr}}{10}}}{2 \\times \\text{sps}}} \times \\text{grad}

    Args:
        x (torch.Tensor): Continuous input signal (BxCxIQxN)
        y (Union[torch.Tensor, int]): The categorical (integer) label for the input
                                      signals.  This can either be a single integer,
                                      which is then assumed to be the label for all
                                      inputs, or it can be a a Tensor (B) specifying a
                                      label for each batch of x.
        input_size (int, optional): Number of time samples that net takes in at a time.
                                    If not provided, it is inferred from the x shape.
                                    Defaults to None.
        net (Model): Classification model to use for computing the gradient signal.
        spr (float): Signal-to-Perturbation ratio (SPR) in dB that is used to scale the
                     power of the perturbation signal crafted and applied.
        sps (int, optional): Samples per symbol (sps) needed to compute the correct
                             scaling for achieving the desired spr. Defaults to 8.

    Raises:
        ValueError: If x is not properly formatted.  Currently only one channel
                    dimension is supported.
        ValueError: If y is an invalid label (negative number) or is provided as a
                    Tensor but the dimensions do not properly match the input, x.

    Returns:
        torch.Tensor: Perturbed signal (x + p) which is formatted as BxCxIQxN

    .. warn::

        This function assumes that Es is 1 when scaling the perturbation to achieve a
        desired SPR.  Therefore, it will first rescale the underlying example to ensure
        that is true.  Generally, this will not cause an issue because the model that
        uses the example will rescale the signal anyways based on its own normalization.

    Reference:
        Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy,
        "Explaining and Harnessing Adversarial Examples",
        https://arxiv.org/abs/1412.6572
    """
    x, y = _convert_or_throw(x=x, y=y)
    input_size = _infer_input_size(x=x, input_size=input_size)
    p = compute_signed_gradient(x=x, y=y, input_size=input_size, sps=sps, net=net)
    p = scale_perturbation(sg=p, spr=spr, sps=sps)

    return x + p


def compute_signed_gradient(
    x: torch.Tensor,
    y: Union[torch.Tensor, int],
    net: Model,
    input_size: int = None,
    sps: int = 8,
) -> torch.Tensor:
    """Compute the signed gradient of a signal, which can later be scaled to achieve an
    untargeted FGSM attack.

    This method does this by:
        - Slicing the signal, x, into discrete examples of length input_size
        - Passing all examples into the neural network (net)
        - Computing the loss function (categorical cross entropy) with respect to the
            true class (y).
        - Backpropagating this to compute the gradient with respect to the input where
            the sign of the gradient can then be computed.

    Args:
        x (torch.Tensor): Continuous input signal (BxCxIQxN)
        y (torch.Tensor): The categorical (integer) label for each of the input signals.
                          This is specified as a Long tensor (B).
        net (Model): Classification model to use for computing the gradient signal.
        input_size (int, optional): Number of time samples that net takes in at a time.
                                    If not provided, it is inferred from the x shape.
                                    Defaults to None.
        sps (int, optional): Samples per symbol that is used for normalizing the signal
                             power before performing the adversarial attack to ensure
                             that the intensity matches what is desired.  Normally, you
                             will not need to provide this as the model should
                             perform normalization itself and therefore undo the linear
                             operation done here.  However, if you're going to use this
                             adversarial example for other purposes, such as determining
                             a bit error rate, then you'll want to ensure this matches
                             your assumptions there.  Defaults to 8.

    Raises:
        ValueError: If x is not properly formatted.  Currently only one channel
                    dimension is supported.
        ValueError: If y is an invalid label (negative number) or is provided as a
                    Tensor but the dimensions do not properly match the input, x.

    Returns:
        torch.Tensor: Sign of the gradient (BxCxIQxN)
    """
    x, y = _convert_or_throw(x=x, y=y)
    input_size = _infer_input_size(x=x, input_size=input_size)
    slicer = Slicer(width=input_size)

    # Ensure that the gradient is tracked at the input, add some noise to avoid any
    # actual zeros in the signal (dithering), and then ensure its the proper shape
    x.requires_grad = True
    _x = _normalize(x=x, sps=sps)
    _x = _dither(_x)
    _x = slicer(_x)

    # Ensure the model is in eval mode so that batch norm/dropout etc. doesn't take
    # effect -- in order to be transparent to the caller we need to restore the state
    # at the end.
    set_training = net.training
    if set_training:
        net.eval()

    # Put the inputs/outputs onto the most probable device that the model is currently
    # on -- this could fail if the model gets split amongst multiple devies, but, that
    # doesn't happen in this code.
    _x = _x.to(net.device)
    y = y.to(net.device)

    # Perform forward/backward pass to get the gradient at the input
    _y = net(_x)
    loss = cross_entropy(_y, y)
    loss.backward()

    # Take the sign of the gradient that can be scaled later
    ret = torch.sign(x.grad.data)

    # Restore the network state so the caller never notices the change
    if set_training:
        net.train()

    return ret


def scale_perturbation(sg: torch.Tensor, spr: float, sps: int = 8) -> torch.Tensor:
    """Scale the signed gradient for an FGSM attack at the specified intensity (spr).

    .. math::

        \\text{grad} = \\text{sign}(\\nabla_X \\mathcalP{L}(f(\\theta, X), y_s))

        p = \\sqrt{\\frac{10^{\\frac{-\\text{spr}}{10}}}{2 \\times \\text{sps}}} \times \\text{grad}

    Args:
        sg (torch.Tensor): Signed gradient, consisting of only +/- 1, that is meant to
                           be linearly scaled to achieve the specified power (spr).
        spr (float): Desired Signal-to-Perturbation ratio in dB.
        sps (int, optional): Samples per symbol which is used for scaling the signed
                             gradient. Defaults to 8.

    .. warn::

        This function assumes that Es is 1 when scaling the perturbation to achieve a
        desired SPR.

    Returns:
        torch.Tensor: Scaled perturbation (Same dimensions as input, sg)
    """
    if spr == np.inf:
        return sg * 0
    multiplier = _compute_multiplier(spr=spr, sps=sps)
    return sg * multiplier
