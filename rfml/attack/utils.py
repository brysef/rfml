"""Utility functions that are common amongst the attacks -- not for external use.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch
from typing import Union, Tuple

# Internal Includes
import rfml.nn.F as F


def _convert_or_throw(
    x: torch.Tensor, y: Union[torch.Tensor, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(x.size()) != 4:
        raise ValueError(
            "PyTorch X Tensor must be of shape BxCxIQxN, not {}".format(x.size())
        )
    elif x.size()[1] != 1:
        raise ValueError("Multiple channel dimensions not currently supported.")
    elif x.size()[2] != 2:
        raise ValueError(
            "The IQ dimension must contain 2 elements, not {}".format(x.size()[2])
        )

    # If y was simply provided by an integer, we can silently transform it into a batch
    # list of labels for the caller
    if isinstance(y, int):
        if y < 0:
            raise ValueError(
                "If providing the label as an integer then it must be positive (or 0). "
                "You gave the label {}".format(y)
            )
        _y = np.zeros((x.shape[0],))
        _y[:] = y
        y = torch.from_numpy(_y.astype(dtype=np.int64))
        # Allow our own checks to go through below for sanity -- even though itd be
        # awkward if we failed our own checks to complain to the caller but its better
        # than a silent failure

    if len(y.size()) != 1:
        raise ValueError("PyTorch Y Tensor must be of shape B, not {}".format(y.size()))
    elif y.size()[0] != x.size()[0]:
        raise ValueError(
            "The batch dimesnions of X and Y must match!  "
            "They were {} and {} respectively.".format(x.size()[0], y.size()[0])
        )
    elif y.dtype != torch.long:
        raise ValueError(
            "The Y data type must be a Long Tensor, not {}".format(y.dtype)
        )

    return x.float(), y.long()


def _infer_input_size(x: torch.Tensor, input_size: int) -> int:
    if input_size is not None:
        return input_size
    time_dim = 3
    return x.size()[time_dim]


def _dither(x: torch.Tensor):
    snr = 100
    voltage = pow(pow(10.0, -snr / 10.0), 0.5)

    noise = x.data.new(x.size()).normal_(0.0, voltage)
    return x + noise


def _normalize(x: torch.Tensor, sps: int) -> torch.Tensor:
    power = F.energy(x, sps=sps)
    # Make the dimensions match because broadcasting is too magical to
    # understand in its entirety... essentially want to ensure that we
    # divide each channel of each example by the sqrt of the power of
    # that channel/example pair
    power = power.view([power.size()[0], power.size()[1], 1, 1])

    return x / torch.sqrt(power)


def _random_uniform_start(x: torch.Tensor, sps: int, spr: int) -> torch.Tensor:
    eps = _compute_multiplier(spr=spr, sps=sps)
    noise = x.data.new(x.size()).uniform_(-eps, eps)
    return x + noise


def _compute_multiplier(spr: Union[float, int], sps: int) -> float:
    multiplier = pow(10, -spr / 10.0)
    multiplier = multiplier / (2 * sps)
    multiplier = pow(multiplier, 0.5)
    return multiplier
