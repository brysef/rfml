"""PyTorch implementation of an IQ Signal Slicer
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Slicer(nn.Module):
    """Turn long continuous signals into discrete examples with a fixed width.

    This can be thought of as batching up discrete examples to perform classification on
    in a real system.  It starts at *offset* and creates as many examples as needed to
    fit all (though it will not create undersized examples so some may be thrown away)
    samples into discrete chunks.  The examples are then concatenated in the batch
    dimension.  The channel and IQ dimensions remain unchanged and naturally the time
    dimension will be identical to *width*.

    This module is differentiable and can therefore be directly integrated in a training
    chain.

    Args:
        width (int): Size of the examples or "number of samples" in the time dimension.
        offset (int, optional): Number of samples to skip at the beginning and end.
                                This can be useful for ignoring filter transients on the
                                sides where the data is unusable.  Defaults to 0.

    Raises:
        ValueError: If width is not a positive integer.
        ValueError: If offset is negative.

    This module assumes that the input is formatted as BxCxIQxT.  The returned output
    from the forward pass will have a large batch dimension and the time dimension will
    match the *width* provided.  The other dimensions are left unchanged.
    """

    def __init__(self, width: int, offset: int = 0):
        if width < 1:
            raise ValueError("Width must be a positive integer, not {}".format(width))
        if offset < 0:
            raise ValueError("Offset cannot be negative -- you gave {}".format(offset))
        super(Slicer, self).__init__()

        self.width = width
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_dim = 0
        time_dim = 3

        n_samples = x.shape[time_dim]

        # Early return as a pass through if the signal is already properly shaped
        if n_samples == self.width:
            return x

        if n_samples < (2 * self.offset + self.width):
            raise ValueError(
                "Not enough samples to perform operation, "
                "input shape={shape}, width={width}, "
                "offset={offset}.".format(
                    shape=x.shape, width=self.width, offset=self.offset
                )
            )

        # First, compute the number of samples and chunks we will end up with
        # Trim off the edges based on offset
        n_samples = n_samples - 2 * self.offset
        # Make sure all examples are evenly sized to width, throwing away the final
        # samples if necessary
        n_chunks = int(np.floor(n_samples / self.width))
        n_samples = int(n_chunks * self.width)

        # Discard the samples outside of the offset ranges
        x = x.narrow(dim=time_dim, start=self.offset, length=n_samples)
        # Create n_chunks from the remaining samples
        # Because we performed the math above, this is ensured to come out to chunks of
        # self.width
        x = x.chunk(chunks=n_chunks, dim=time_dim)

        # Now that we have a list of examples, concatenate them in the batch dimension
        x = torch.cat(x, dim=batch_dim)

        return x
