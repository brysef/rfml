"""PyTorch implementation of center/carrier frequency offset.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# PyTorch Includes
import torch
import torch.nn as nn

# External Includes
import numpy as np


class CFO(nn.Module):
    """Center Frequency Offset Channel model implemented in PyTorch.

    A center frequency offset receiver effect can be simulated by

    .. math::

        s_{\\text{rx}}(t) = e^{-j 2 \pi f_0 t} s_{\\text{tx}}(t)

    Where :math:`f_0` represents the normalized frequency offset in terms of the sample
    rate.

    Further, a complex number, which :math:`s_{\\text{tx}}` is a vector of, is
    represented as

    .. math::

        z = a + j b

    where :math:`a` represents the real portion and :math:`b` represents the imaginary
    portion of the number.

    Due to Euler's identity, a complex sine wave can be represented using

    .. math::

        e^{-j 2 \pi f_o t} = \operatorname{cos}(2 \pi f_0 t) + j \operatorname{sin}(2 \pi f_0 t)

    Therefore,

    .. math::

        \\begin{aligned}
            z_2 &= z_1 \\times e^{-j 2 \pi f_0 t}

                &= [a + jb] \\times [\operatorname{cos}(2 \pi f_0 t) + j \operatorname{sin}(2 \pi f_0 t)]

                &= a \operatorname{cos}(2 \pi f_0 t) + a j \operatorname{sin}(2 \pi f_0 t) + b j \operatorname{cos}(2 \pi f_0 t) + b j^2 \operatorname{sin}(2 \pi f_0 t)

                &= a \operatorname{cos}(2 \pi f_0 t) + a j \operatorname{sin}(2 \pi f_0 t) + b j \operatorname{cos}(2 \pi f_0 t) - b \operatorname{sin}(2 \pi f_0 t)

                &= [a \operatorname{cos}(2 \pi f_0 t) - b \operatorname{sin}(2 \pi f_0 t)] + j[a \operatorname{sin}(2 \pi f_0 t) + b \operatorname{cos}(2 \pi f_0 t)]
        \end{aligned}

    Args:
        cfo (float, optional): Center frequency offset percentage normalized to sample
                               rate.  This can be updated by calling *set_cfo*.
                               Defaults to 0.0.

    Raises:
        ValueError: If the provided frequency offset is not in [-0.5, 0.5].

    This module assumes that the input tensor is in BxCxIQxN format and returns a Tensor
    with the same dimensions.
    """

    def __init__(self, cfo: float = 0.0):
        if np.abs(cfo) > 0.5:
            raise ValueError(
                "The center frequency offset must be between -0.5 and 0.5 of the "
                "sample rate"
            )
        super(CFO, self).__init__()
        self.cfo = cfo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optimization: Avoid computing a complex sine wave if the freq is 0
        # because this would just result in multiplying the signal by 1 anyways
        if self.cfo == 0.0:
            return x

        t = x.data.new(torch.arange(x.size()[3], dtype=x.dtype, device=x.device))
        t = t * 2.0 * np.pi * self.cfo

        cos = torch.cos(t)
        sin = torch.sin(t)

        iq_dim = 2
        a, b = x.chunk(chunks=2, dim=iq_dim)

        r = a * cos - b * sin
        c = a * sin + b * cos

        x = torch.cat((r, c), dim=iq_dim)

        return x

    def set_cfo(self, cfo: float):
        """Set the normalized center frequency offset to be used on the next pass.

        Args:
            cfo (float): Center frequency offset percentage normalized to sample rate.

        Raises:
            ValueError: If the provided frequency offset is not in [-0.5, 0.5].
        """
        if np.abs(cfo) > 0.5:
            raise ValueError(
                "The center frequency offset must be between -0.5 and 0.5 of the "
                "sample rate"
            )
        self.cfo = cfo
