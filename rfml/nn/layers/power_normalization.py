"""Normalize the power across each example/channel to 1.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch
import torch.nn as nn

# Internal Includes
import rfml.nn.F as F


class PowerNormalization(nn.Module):
    """Perform average energy per sample (power) normalization.

    Power Normalization would be performed as follows for each batch/channel:

    .. math::

        x = \\frac{x}{\sqrt{\mathbb{E}[x]}}

    This module assumes that the signal is structured as:

    .. math::

        Batch x Channel x IQ x Time.

    Where the power normalization is performed along the T axis using the power
    measured in the complex valued I/Q dimension.

    The outputs of this layer match the inputs:

    .. math::

        Batch x Channel x IQ x Time
    """

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(
                "The inputs to the PowerNormalization layer must have 4 dimensions "
                "(BxCxIQxT), input shape was {}".format(x.shape)
            )
        if x.shape[2] != 2:
            raise ValueError(
                "The inputs to the PowerNormalization layer must be 'complex valued' "
                "by having 2 elements in the IQ dimension (BxCxIQxT), input shape was "
                "{}".format(x.shape)
            )

        energy = F.energy(x)
        # Make the dimensions match because broadcasting is too magical to
        # understand in its entirety... essentially want to ensure that we
        # divide each channel of each example by the sqrt of the power of
        # that channel/example pair
        energy = energy.view([energy.size()[0], energy.size()[1], 1, 1])

        return x / torch.sqrt(energy)
