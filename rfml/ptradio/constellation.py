"""PyTorch implementation of (un)mapping symbol constellations.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np

import torch
import torch.nn as nn

# Internal Includes
from rfml.nn.F import evm


class ConstellationMapper(nn.Module):
    """Layer that transforms "chunks" into symbols in the forward pass.

    Args:
        constellation (np.ndarray): List of complex valued points in the constellation
                                    (2xM numpy array where M is the order of the
                                    modulation e.g. 2 for BPSK, 4 for QPSK, etc.)

    Raises:
        ValueError: If the constellation does not match the expected 2xM shape.

    The forward pass should include Long Tensors that are in the half-open interval
    [0-M).  The output of the forward pass will then be the symbol at that index in the
    constellation.

    This module assumes that the input is [N] and extends the output to be of the format
    [BxCxIQxN] where B and C are 1, IQ is 2, and N matches the input.
    """

    def __init__(self, constellation: np.ndarray):
        super(ConstellationMapper, self).__init__()
        if constellation.shape[0] != 2 or constellation.shape[1] < 2:
            raise ValueError(
                "Expected the constellation to be complex valued (2xM), not "
                "({})".format(constellation.shape)
            )

        self.constellation = torch.Tensor(constellation.astype(np.float64))
        # Creating a parameter ensures that it gets placed onto the GPU with cuda()
        self.constellation = nn.Parameter(data=self.constellation, requires_grad=False)

    def get_M(self) -> int:
        """Return the number of symbols in the constellation/order of the modulation
        """
        # Note, these can't be collapsed into a parent class because the constellation
        # dimensions are different between map/unmap
        return self.constellation.shape[1]

    def get_bps(self) -> float:
        """Return the number of bits per symbol

        .. warning::

            This could be fractional if M is not a power of 2.
        """
        return np.log2(self.get_M())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Select the desired symbol from the constellation
        x = torch.index_select(input=self.constellation, dim=1, index=x)
        # Add in batch/channel dimmensions
        x = x.unsqueeze(dim=0)
        x = x.unsqueeze(dim=0)
        return x


class ConstellationUnmapper(nn.Module):
    """Layer that transforms symbols into "chunks" in the forward pass via a nearest
    neighbors algorithm.

    Args:
        constellation (np.ndarray): List of complex valued points in the constellation
                                    (2xM numpy array where M is the order of the
                                    modulation e.g. 2 for BPSK, 4 for QPSK, etc.)

    The forward pass should provide complex valued symbols and the output will be Long
    Tensors that are in the half-open interval [0-M).

    This module assumes that the input is [BxCxIQxN] and assumes that B and C are both
    1.  Therefore, the output is provided as [N].
    """

    def __init__(self, constellation: np.ndarray):
        super(ConstellationUnmapper, self).__init__()
        if constellation.shape[0] != 2 or constellation.shape[1] < 2:
            raise ValueError(
                "Expected the constellation to be complex valued (2xM), not "
                "({})".format(constellation.shape)
            )

        self.constellation = torch.Tensor(constellation.astype(np.float32))
        # Add in batch/channel dimmensions
        self.constellation.unsqueeze_(dim=0)
        self.constellation.unsqueeze_(dim=0)
        # Add in an single time dimension that will be used for broadcasting in EVM
        self.constellation.unsqueeze_(dim=3)
        # Creating a parameter ensures that it gets placed onto the GPU with cuda()
        self.constellation = nn.Parameter(data=self.constellation, requires_grad=False)

    def get_M(self) -> int:
        """Return the number of symbols in the constellation/order of the modulation
        """
        # Note, these can't be collapsed into a parent class because the constellation
        # dimensions are different between map/unmap
        return self.constellation.shape[4]

    def get_bps(self) -> float:
        """Return the number of bits per symbol

        .. warning::

            This could be fractional if M is not a power of 2.
        """
        return np.log2(self.get_M())

    def forward(self, x):
        time_dim = 3
        # Create an addition dimension for selecting the index
        x = x.unsqueeze(dim=-1)
        # Expand the dimension to match the number of symbols we're going to compare to
        sizes = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            self.constellation.shape[time_dim],
        )
        x.expand(sizes)
        x = evm(x, self.constellation)
        # Select the index of the closest symbol in the constellation to the received
        # symbol estimate
        x = x.argmin(dim=4, keepdim=False)
        return x
