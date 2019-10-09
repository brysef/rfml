"""Data (generation) helpers to simplify the code flow of Jupyter notebooks.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np

import torch

from typing import List, Tuple

# Internal Includes
from rfml.data import Encoder
from rfml.ptradio import Slicer, Transmitter


def generate_dataset(
    n_examples: int,
    le: Encoder,
    input_samples: int = 128,
    modulations: List[str] = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"],
    sps: int = 8,
    half_filter_span: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sps = 8
    half_filter_span *= sps
    n_symbols = int((n_examples * input_samples) / sps)

    x = list()  # Inputs
    y = list()  # Categorical Class Labels

    # Utilize the Slicer object to turn long/continuous signals
    # e.g. 1x1x2x80000 into "batched" windowed views of the signal
    # e.g. 1000x1x2x128 which the DNN takes in
    slicer = Slicer(offset=half_filter_span, width=input_samples)

    for modulation in ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"]:
        print("Generating data for {modulation}".format(modulation=modulation))
        print("==============================")
        tx = Transmitter(modulation=modulation)

        iq = slicer(tx.modulate(n_symbols=n_symbols))
        print("IQ Shape: {}".format(iq.shape))
        x.append(iq)

        label = torch.from_numpy(np.array(le.encode([modulation] * iq.shape[0])))
        print("Label Shape: {}".format(label.shape))
        y.append(label)

        print("==============================\n")

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    return x, y
