"""Verify that the PyTorch calculation of EVM matches numpy.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import torch
from unittest import TestCase

# Internal Includes
import rfml.nn.F as F


class TestEVM(TestCase):
    def test_operation(self):
        """Verify that a basic EVM calculation matches numpy (no batch).
        """
        n_trials = 100
        time_dim = 3

        for _ in np.arange(n_trials):
            x = _create_signal()  # 1d array of complex samples
            y = x + np.random.normal(size=x.shape) + 1j * np.random.normal(size=x.shape)
            torch_x = _signal_to_torch(x)  # [BxCxIQxT] Tensor
            torch_y = _signal_to_torch(y)  # [BxCxIQxT] Tensor

            expected_evm = _numpy_evm(x, y)  # 1d array of complex samples
            actual_evm = F.evm(torch_x, torch_y)  # [BxCxF]

            self.assertEqual(actual_evm.shape, (1, 1, 1, torch_x.shape[time_dim]))
            actual_evm = actual_evm.detach().numpy()[0, 0, :]
            np.testing.assert_allclose(actual_evm, expected_evm, rtol=1e-9, atol=0)

    def test_batchoperation(self):
        """Verify that EVM can be calculated in parallel through the batch dimension.
        """
        n_trials = 10
        time_dim = 3

        for _ in np.arange(n_trials):
            n_batch = np.random.randint(low=128, high=512)
            n_time = np.random.randint(low=64, high=1028)

            # Create a list of signals in the batch dimension
            expected_evm = list()
            torch_x = list()
            torch_y = list()
            for _ in np.arange(n_batch):
                _x = _create_signal(n_time=n_time)  # 1d array of complex samples
                _y = (
                    _x
                    + np.random.normal(size=_x.shape)
                    + 1j * np.random.normal(size=_x.shape)
                )
                _expected_evm = _numpy_evm(_x, _y)  # 1d array of complex samples
                _torch_x = _signal_to_torch(_x)  # [BxCxIQxT] Tensor
                _torch_y = _signal_to_torch(_y)  # [BxCxIQxT] Tensor

                expected_evm.append(_expected_evm)
                torch_x.append(_torch_x)
                torch_y.append(_torch_y)

            torch_x = torch.cat(torch_x)
            torch_y = torch.cat(torch_y)

            actual_evm = F.evm(torch_x, torch_y)  # [BxCxF]

            self.assertEqual(actual_evm.shape, (n_batch, 1, 1, torch_x.shape[time_dim]))
            actual_evm = actual_evm.detach().numpy()[:, 0, :]
            np.testing.assert_allclose(actual_evm, expected_evm, rtol=1e-9, atol=0)

    def test_invalidinputs(self):
        """Verify that the function protects itself against bad inputs.
        """
        # No batch/channel dimension
        with self.assertRaises(ValueError):
            F.evm(torch.zeros(2, 1024), torch.zeros(2, 1024))

        # No batch dimension
        with self.assertRaises(ValueError):
            F.evm(torch.zeros(1, 2, 1024), torch.zeros(1, 2, 1024))

        # Incorrect IQ dimension
        with self.assertRaises(ValueError):
            F.evm(torch.zeros(100, 1, 1, 1024), torch.zeros(100, 1, 1, 1024))

        # Mismatching shapes
        with self.assertRaises(ValueError):
            F.evm(torch.zeros(100, 5, 1, 1024), torch.zeros(100, 5, 1, 512))


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


def _numpy_evm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ret = x - y  # Error Vector
    ret = np.abs(ret)  # Magnitude

    ret = ret[np.newaxis, :]  # Add in an empty batch dimension

    return ret
