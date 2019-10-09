"""PyTorch implementation of a Root Raised Cosine filter
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RRC(nn.Module):
    """Root Raised Cosine filter implemented as a PyTorch convolution.

    Args:
        alpha (float, optional): Roll-off factor of the filter. Defaults to 0.35.
        sps (int, optional): Samples per symbol. Defaults to 8.
        filter_span (int, optional): One-sided filter span in number of symbols.
                                     Defaults to 8.
        add_pad (bool, optional): True if padding should be added to simulate a tap
                                  delay.  This should be True when this module is used
                                  as a pulse shaping filter and False when this module
                                  is used as a matched filter (because the extra data
                                  is useless anyways). Defaults to False.

    Raises:
        ValueError: If alpha is not in the interval (0.0, 1.0)
        ValueError: If sps is not 2 or more
        ValueError: If filter_span is not a positive integer
    """

    def __init__(
        self,
        alpha: float = 0.35,
        sps: int = 8,
        filter_span: int = 8,
        add_pad: bool = False,
    ):
        super(RRC, self).__init__()
        self.impulse_response = torch.tensor(
            self._get_impulse_response(alpha, sps, filter_span).astype(np.float32)
        )
        # Creating a parameter ensures that it gets placed onto the GPU with cuda()
        self.impulse_response = nn.Parameter(
            data=self.impulse_response, requires_grad=False
        )

        # Add in out_channels x in_channels/groups x H x W
        # 1 x 1 x 1 x nTaps
        self.impulse_response.unsqueeze_(dim=0)
        self.impulse_response.unsqueeze_(dim=0)
        self.impulse_response.unsqueeze_(dim=0)

        self.pad = sps * filter_span
        self.n_taps = 2 * filter_span * sps + 1
        self.add_pad = add_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_pad:
            x = F.pad(x, [self.n_taps - 1, self.n_taps], "constant", 0)
            x = F.conv2d(x, self.impulse_response, padding=0)
        else:
            x = F.conv2d(x, self.impulse_response, padding=[0, self.pad])
        return x

    def _get_impulse_response(self, alpha: float, sps: int, filter_span: int):
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("Alpha must be between (0, 1.0), not {}".format(alpha))
        if sps < 2:
            raise ValueError("Sps must be 2 or more, not {}".format(sps))
        if filter_span <= 0:
            raise ValueError(
                "Filter span must be a positive integer, not {}".format(filter_span)
            )

        # EQN for the impulse response is taken from NASA slide 11 -- others exist
        # https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120008631.pdf
        filter_len = 2 * filter_span * sps + 1
        impulse_response = np.zeros((filter_len))

        for t in np.arange(-filter_span * sps, filter_span * sps + 1):
            i = t + (filter_span * sps)
            if t == 0:
                impulse_response[i] = 1 + alpha * (4 / np.pi - 1)
                impulse_response[i] *= 1.0 / np.sqrt(sps)
            else:
                impulse_response[i] = np.sin((1 - alpha) * np.pi * t / sps)
                impulse_response[i] /= 4 * alpha * t / sps
                impulse_response[i] += np.cos((1 + alpha) * np.pi * t / sps)
                impulse_response[i] *= 2.0 * alpha / (np.pi * np.sqrt(sps))
                impulse_response[i] /= 1.0 - (4.0 * alpha * t / sps) ** 2

                # Not sure why this extra gain is needed to match the impulse response
                # of GNU Radio filter designer -- but it matches exactly with this.  It
                # likely has to do with treating the filter as complex.
                impulse_response[i] *= 2.0

        return impulse_response
