from rfml.ptradio import RRC, Upsample, Downsample
from rfml.ptradio.modem import _qpsk_constellation
from rfml.nn.F import evm

import numpy as np

import torch
from torch.nn import Sequential, Parameter
from torch.autograd import Variable
from torch.optim import SGD

n_symbols = 32
indices = np.random.randint(low=0, high=4, size=n_symbols)
target_symbols = np.array([_qpsk_constellation[i] for i in indices])
target_symbols = np.stack((target_symbols.real, target_symbols.imag))
_target_symbols = torch.from_numpy(
    target_symbols[np.newaxis, np.newaxis, ::].astype(np.float32)
)

mean = torch.zeros((1, 1, 2, _target_symbols.shape[3]))
std = torch.ones((1, 1, 2, _target_symbols.shape[3]))
tx_symbols = torch.nn.Parameter(torch.normal(mean, std))

optimizer = SGD((tx_symbols,), lr=10e-2, momentum=0.9)

tx_chain = Sequential(
    Upsample(i=8), RRC(alpha=0.35, sps=8, filter_span=8, add_pad=True)
)
rx_chain = Sequential(
    RRC(alpha=0.35, sps=8, filter_span=8, add_pad=False), Downsample(offset=8 * 8, d=8)
)

n_epochs = 151
for i in range(n_epochs):
    tx_signal = tx_chain(tx_symbols)
    rx_symbols = rx_chain(tx_signal)
    loss = torch.mean(evm(rx_symbols, _target_symbols))

    if i % 15 == 0:
        print("Loss @ epoch {}: {:3f}".format(i, loss))

    loss.backward()
    optimizer.step()
    tx_symbols.grad.zero_()
