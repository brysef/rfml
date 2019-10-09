from rfml.nn.F import psd
from rfml.ptradio import RRC

import numpy as np

import torch
from torch.nn import Parameter
from torch.optim import SGD

n_time = 1024

# Create a white gaussian noise signal -- therefore ~ flat across frequency
mean = torch.zeros((1, 1, 2, n_time))
std = torch.ones((1, 1, 2, n_time)) / 25.0
signal = torch.nn.Parameter(torch.normal(mean, std))
t = np.arange(n_time)

# Define our "target" PSD profile to be the spectrum of the root raised cosine
rrc = RRC()
impulse = rrc.impulse_response
# The impulse response is real valued so we'll make it "complex" by just adding
# another dimension in for IQ and setting the imaginary portion to 0
impulse = torch.cat((impulse, impulse), dim=2)
impulse[:, :, 1, :] = 0.0

# In order to match dimensions with our desired frequency resolution by
# setting n_time to be the FFT length -- we must pad with some zeros
_to_pad = torch.zeros(
    (impulse.shape[0], impulse.shape[1], impulse.shape[2], n_time - impulse.shape[3])
)
impulse = torch.cat((impulse, _to_pad), dim=3)

target_psd = psd(impulse)

optimizer = SGD((signal,), lr=50e-4, momentum=0.9)

n_epochs = 151
for i in range(n_epochs):
    cur_psd = psd(signal)
    loss = torch.mean((cur_psd - target_psd) ** 2)

    if i % 15 == 0:
        print("Loss @ epoch {}: {:3f}".format(i, loss))

    loss.backward()
    optimizer.step()
    signal.grad.zero_()
