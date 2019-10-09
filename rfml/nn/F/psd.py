"""Calculate the power spectral density (PSD) of an input signal.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch

# Internal Includes
from .energy import energy


def psd(x: torch.Tensor) -> torch.Tensor:
    # TODO allow the user to specify the number of FFT points in the future
    #      thus, we'd have to chop up or pad the signal here in an intelligent manner
    channel_dim = 1
    iq_dim = 2
    time_dim = 3

    if len(x.size()) != 4:
        raise ValueError(
            "The input Tensor must be of shape [BxCxIQxT] -- "
            "your input had shape {}".format(x.size())
        )
    if x.size()[channel_dim] != 1:
        # TODO handle multiple channels more intelligently
        raise ValueError(
            "The input tensor must only contain a single channel -- "
            "your's contained {}".format(x.size()[channel_dim])
        )
    if x.size()[iq_dim] != 2:
        raise ValueError(
            "The input Tensor must be of shape [BxCxIQxT] -- "
            "your input had shape {} which doesn't have a dimensioniality of 2"
            "for the IQ dimension".format(x.size())
        )

    # PyTorch FFT takes the Tensor as BxTxIQ for a 1d FFT whereas this library
    # represents data as BxCxIQxT -- for now, we've enforced above that there is only
    # one dimension and thus this channel dimension can be removed and later added back
    # in.  We can then tranpose the IQ and time dimensions to match the expectation from
    # PyTorch's FFT and later undo that operation as well.
    x = torch.transpose(x, dim0=iq_dim, dim1=time_dim)
    x = x.squeeze(dim=channel_dim)

    x = torch.fft(x, signal_ndim=1)

    x = x.unsqueeze(dim=channel_dim)
    x = torch.transpose(x, dim0=iq_dim, dim1=time_dim)

    # Now we have a list of complex numbers -- we only need their power so one dimension
    # collapses during the magnitude^2 calculation
    r, c = x.chunk(chunks=2, dim=iq_dim)
    psd = (r * r) + (c * c)  # power is magnitude squared so sqrt cancels
    psd = psd.squeeze(dim=iq_dim)

    return psd
