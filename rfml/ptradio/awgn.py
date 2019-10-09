"""PyTorch implementation of an AWGN wireless channel
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch
import torch.nn as nn
import torch.nn.functional as F


class AWGN(nn.Module):
    """Additive White Gaussian Noise (AWGN) Channel model implemented in PyTorch.

    The noise power is provided by SNR, which can be updated by calling *set_snr*.
    Each forward pass will have a different noise realization but the same SNR (as long
    as it has not been changed).  This layer has no effect on sizes and can be made to
    be a pass through by setting SNR to None.

    Args:
        snr (float, optional): Signal-to-Noise ratio.  This can be overriden during
                               operation by calling set_snr.  Defaults to None.

    .. warning::

        This layer assumes that the average energy per symbol of the underlying signal
        is 1 (0 dB) when calculating the noise power.

    This module makese no assumptions about the shape of the input and returns an
    identically shaped output.
    """

    def __init__(self, snr: float = None):
        super(AWGN, self).__init__()
        self.snr = snr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.snr is None:
            return x
        noise = x.data.new(x.size()).normal_(0.0, self._calculate_noise_voltage())
        return x + noise

    def set_snr(self, snr: float):
        """Set the signal to noise ratio in dB"""
        self.snr = snr

    def _calculate_noise_voltage(self) -> float:
        if self.snr is None:
            return 0
        # The power has to be divided by two here because the PyTorch noise
        # generation is not "complex valued" and therefore we have to split the
        # power evenly between I and Q
        return pow(pow(10.0, -self.snr / 10.0) / 2.0, 0.5)
