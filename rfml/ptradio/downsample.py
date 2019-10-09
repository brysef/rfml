"""PyTorch implementation of downsampling.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# PyTorch Includes
import torch
import torch.nn as nn

# External Includes
import numpy as np


class Downsample(nn.Module):
    """PyTorch downsampling implementation.

    Args:
        offset (int, optional): Transient samples at the beginning and end of the signal
                                to leave off of the output (e.g. filter transients).
                                Defaults to 0.
        d (int, optional): Decimation factor -- only integer valued sample rate
                           conversions are allowed. Defaults to 8.

    Raises:
        ValueError: If offset is less than 0.
        ValueError: If d is less than 2.

    This module assumes that the time dimension is 3 (e.g. [BxCxIQxN]).  It returns a
    tensor in the same format, but, the output has been downsampled in the time
    dimension (e.g. [BxCxIQxN/D] ignoring any provided offsets).
    """

    def __init__(self, offset: int = 0, d: int = 8):
        if offset < 0:
            raise ValueError("Offset must be positive, not {}".format(offset))
        if d < 2:
            raise ValueError(
                "You must decimate by at least a factor of 2, not {}".format(d)
            )
        super(Downsample, self).__init__()
        self.offset = offset
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_dimension = 3
        # Estimate the number of "symbols" -- which, speaking more broadly is simply the
        # number of samples that will be in the output array.  The word symbols is used
        # here for a variable simply because the primary use of this code is to
        # downsample to one sample per symbol after a matched filter output.
        n_samples = x.size()[time_dimension]
        n_symbols = (n_samples - 2 * self.offset + 1) / self.d - 1
        # Generate indexes to simply select every "dth" sample
        indices = torch.tensor(
            np.arange(n_symbols) * self.d + self.offset, dtype=torch.long
        )
        indices = indices.to(device=x.device)
        # Downsample the signal by only pulling out the specified indexes
        x = x.index_select(dim=time_dimension, index=indices)
        return x
