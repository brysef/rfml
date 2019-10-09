"""PyTorch implementation of Error Vector Magnitude
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch


def evm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Compute the Error Vector Magnitude (EVM) between two signals.

    .. math::

        \operatorname{EVM}(s_{rx}, s_{tx}) = \sqrt{\mathbb{R}(s_{rx} - s_{tx})^2 + \mathbb{C}(s_{rx} - s_{tx})^2}

    Args:
        x (torch.Tensor): First signal (BxCxIQxN)
        y (torch.Tensor): Second signal (BxCxIQxN)

    Returns:
        [torch.Tensor]: Error Vector Magnitude per sample (BxCx1xN).  Note that the
                        returned signal is no longer *complex* as it is only a
                        magnitude and therefore only has a dimension size of 1 where IQ
                        used to be.
    """
    iq_dim = 2
    if len(x.size()) < 4:
        raise ValueError(
            "The inputs to EVM must have at least four dimensions ([BxCxIQxT]) -- "
            "your x input had shape {}".format(x.size())
        )
    if len(y.size()) < 4:
        raise ValueError(
            "The inputs to EVM must have at least four dimensions ([BxCxIQxT]) -- "
            "your y input had shape {}".format(y.size())
        )
    # Including these error checks prevents the user from using broadcasting
    # I'm not sure of a a good way to check for broadcastable shapes other than just
    # letting it fail and throw its own error message --
    # This ruins a lot of the defensive programming that can be done for this function
    # if len(x.size()) != len(y.size()):
    #    raise ValueError(
    #        "The shapes of x and y must match!  "
    #        "x shape ({}) -- y shape ({})".format(x.size(), y.size())
    #    )
    # for i, (xdim, ydim) in enumerate(zip(x.size(), y.size())):
    #    if xdim != ydim:
    #        raise ValueError(
    #            "The shapes of x and y must match!  They differ at location {}.  "
    #            "x shape ({}) -- y shape ({})".format(i, x.size(), y.size())
    #        )
    if x.size()[iq_dim] != 2:
        raise ValueError(
            "The input shapes must be *complex*, e.g. they must have a dimension size "
            "of 2 at location {} but x has dimensions {}".format(iq_dim, x.size())
        )
    if y.size()[iq_dim] != 2:
        raise ValueError(
            "The input shapes must be *complex*, e.g. they must have a dimension size "
            "of 2 at location {} but y has dimensions {}".format(iq_dim, y.size())
        )

    # Subtract the two signals element wise to get an error vector
    ev = x - y

    # Compute the per sample magnitude
    rev, cev = ev.chunk(chunks=2, dim=iq_dim)
    evm = torch.sqrt(rev * rev + cev * cev)

    return evm
