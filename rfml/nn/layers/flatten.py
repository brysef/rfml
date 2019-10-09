"""Flatten signals/images into features.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch
import torch.nn as nn


class Flatten(nn.Module):
    """Flatten the channel, IQ, and time dims into a single feature dim.

    This module assumes that the input signal is structured as:

    .. math::

        Batch x Channel x IQ x Time

    Args:
        preserve_time (bool, optional): If provided as True then the time dimension is
                                        preserved in the outputs and only the IQ and
                                        Channel dimensions are concatenated together.
                                        Otherwise, the time dimension is also collapsed
                                        to form a single feature dimension.  Generally,
                                        you will set this to False if the layer after
                                        Flatten will be a Linear layer and set this to
                                        True if the layer after Flatten will be a
                                        Recurrent layer that utilizes the time
                                        dimension.  Defaults to False.

    The outputs of this layer, if *preserve_time* is not set to True, are:

    .. math::

        Batch x Features

    Where features is the product of the flattened dimensions:

    .. math::

        (Channel x IQ x Time)

    The outputs of this layer, if *preserve_time* is set to True, are:

    .. math::

        Batch x Time x Features

    Where features is the product of the flattened dimensions:

    .. math::

        (Channel x IQ)
    """

    def __init__(self, preserve_time: bool = False):
        super().__init__()
        self._preserve_time = preserve_time

    def forward(self, x: torch.Tensor):
        if self._preserve_time:
            return self._flatten_preserve_time(x=x)
        else:
            return self._flatten(x=x)

    def _flatten(self, x: torch.Tensor):
        if len(x.shape) < 2:
            raise ValueError(
                "The inputs to the Flatten layer must have at least 2 dimensions (e.g. "
                "BxCxIQxT), input shape was {}".format(x.shape)
            )
        # It doesn't entirely matter how many dimensions are in the input or if it is
        # properly structured as 'complex valued' (IQ dimension has 2 values).
        # Therefore, the code to implement this is more general while leaving the
        # docstring more explicit to avoid confusing a caller reading the documentation.
        x = x.contiguous()
        x = x.view(x.size()[0], -1)
        return x

    def _flatten_preserve_time(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(
                "The inputs to the Flatten layer must have at least 4 dimensions (e.g. "
                "BxCxIQxT), input shape was {}".format(x.shape)
            )
        channel_dim, time_dim = 1, 3

        # BxCxIQxT
        x = x.transpose(channel_dim, time_dim)
        # BxTxCxIQ -- Can now collapse the final two dimensions
        x = x.contiguous()
        x = x.view(x.size()[0], x.size()[1], -1)
        return x
