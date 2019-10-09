"""Calculate the average energy (per symbol if provided) for each example.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch


def energy(x: torch.Tensor, sps: float = 1.0):
    """Calculate the average energy (per symbol if provided) for each example.

    This function assumes that the signal is structured as:

    .. math::

        Batch x Channel x IQ x Time.

    Args:
        x (torch.Tensor): Input Tensor (BxCxIQxT)
        sps (int, optional): Samples per symbol, essentially the power is multiplied by
                             this value in order to calculate average energy per symbol.
                             Defaults to 1.0.

    .. math::

        \mathbb{E}[E_{s}] = \\frac{\\text{sps}}{N} \sum_{i=0}^{N} |s_i|^2

        |s_i| = \sqrt{\mathbb{R}^2 + \mathbb{C}^2}

    Returns:
        [torch.Tensor]: Average energy per example per channel (BxC)
    """
    if len(x.shape) != 4:
        raise ValueError(
            "The inputs to the energy function must have 4 dimensions (BxCxIQxT), "
            "input shape was {}".format(x.shape)
        )
    if x.shape[2] != 2:
        raise ValueError(
            "The inputs to the energy function must be 'complex valued' by having 2 "
            "elements in the IQ dimension (BxCxIQxT), input shape was {}".format(
                x.shape
            )
        )
    iq_dim = 2
    time_dim = 3

    r, c = x.chunk(chunks=2, dim=iq_dim)
    power = (r * r) + (c * c)  # power is magnitude squared so sqrt cancels

    # pylint: disable=no-member
    # The linter isn't able to find the "mean" function but its there!
    x = torch.mean(power, dim=time_dim) * sps

    # This Tensor still has an unnecessary singleton dimensions in IQ
    x = x.squeeze(dim=iq_dim)

    return x
