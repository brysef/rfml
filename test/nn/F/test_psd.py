"""Verify that the PyTorch calculation of PSD is correct.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import torch
from unittest import TestCase

# Internal Includes
import rfml.nn.F as F


class TestPSD(TestCase):
    def test_operation(self):
        """Verify that a basic power calculation matches numpy (no batch).
        """
        n_trials = 100
        time_dim = 3

        for _ in np.arange(n_trials):
            s = _create_signal()  # 1d array of complex samples
            torch_s = _signal_to_torch(s)  # [BxCxIQxT] Tensor

            expected_psd = _numpy_psd(s)  # 1d array of complex samples
            actual_psd = F.psd(torch_s)  # [BxCxF]

            self.assertEqual(actual_psd.shape, (1, 1, torch_s.shape[time_dim]))
            actual_psd = actual_psd.detach().numpy()[0, 0, :]
            np.testing.assert_allclose(actual_psd, expected_psd, rtol=1e-9, atol=0)

    def test_batchoperation(self):
        """Verify that the PSD can be performed in parallel through the batch dimension.
        """
        n_trials = 10
        time_dim = 3

        for _ in np.arange(n_trials):
            n_batch = np.random.randint(low=128, high=512)
            n_time = np.random.randint(low=64, high=1028)

            # Create a list of signals in the batch dimension
            expected_psd = list()
            torch_s = list()
            for _ in np.arange(n_batch):
                _s = _create_signal(n_time=n_time)  # 1d array of complex samples
                _expected_psd = _numpy_psd(_s)  # 1d array of complex samples
                _torch_s = _signal_to_torch(_s)  # [BxCxIQxT] Tensor

                expected_psd.append(_expected_psd)
                torch_s.append(_torch_s)

            torch_s = torch.cat(torch_s)

            actual_psd = F.psd(torch_s)  # [BxCxF]

            self.assertEqual(actual_psd.shape, (n_batch, 1, torch_s.shape[time_dim]))
            actual_psd = actual_psd.detach().numpy()[:, 0, :]
            np.testing.assert_allclose(actual_psd, expected_psd, rtol=1e-9, atol=0)

    def test_invalidinputs(self):
        """Verify that the function protects itself against bad inputs.
        """
        # No batch/channel dimension
        with self.assertRaises(ValueError):
            F.psd(torch.zeros(2, 1024))

        # No batch dimension
        with self.assertRaises(ValueError):
            F.psd(torch.zeros(1, 2, 1024))

        # Incorrect IQ dimension
        with self.assertRaises(ValueError):
            F.psd(torch.zeros(100, 1, 1, 1024))

        # Multiple Channels
        with self.assertRaises(ValueError):
            F.psd(torch.zeros(100, 5, 1, 1024))


def _create_signal(n_time: int = None) -> np.ndarray:
    # Generate a random signal as the sum of complex sine waves with random amplitude,
    # phase, and frequency
    n_signals = np.random.randint(low=1, high=1024 + 1)
    if n_time is None:
        n_time = np.random.randint(low=124, high=1024 + 1)

    t = np.arange(n_time)[:, np.newaxis]
    fc = np.random.lognormal(size=(1, n_signals))
    theta = np.random.uniform(low=0.0, high=2 * np.pi, size=(1, n_signals))
    a = np.random.lognormal(size=(1, n_signals))

    s = a * np.exp(1j * fc * t + theta)
    s = s.sum(axis=1)  # Sum all of the signals and leave the time dimension

    return s


def _signal_to_torch(s: np.ndarray) -> torch.Tensor:
    iq = np.stack((s.real, s.imag))
    ret = torch.from_numpy(iq)  # [IQxT]

    # Add batch and channel dimensions -- [BxCxIQxT]
    ret.unsqueeze_(dim=0)
    ret.unsqueeze_(dim=0)

    return ret


def _numpy_psd(s: np.ndarray) -> np.ndarray:
    ret = np.fft.fft(s)  # Take the FFT, which is a complex value
    ret = np.abs(ret) ** 2  # Convert to power

    return ret
