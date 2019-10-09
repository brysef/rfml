"""Verify that the PyTorch calculation of energy is correct.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import torch
from unittest import TestCase

# Internal Includes
import rfml.nn.F as F


class TestEnergy(TestCase):
    def test_operation(self):
        """Verify that a basic power calculation matches numpy (no batch).
        """
        n_trials = int(10e3)

        def _get_numpy_and_torch_energy(fc, a, num=1024):
            # Create a complex valued sine wave using numpy
            t = np.linspace(0.0, 100.0, num=num)  # Time
            s = a * np.exp(2 * np.pi * 1j * fc * t)  # Output is complex
            iq = np.stack((s.real, s.imag))

            # Convert to PyTorch in order to use the PyTorch functions
            # Add singleton batch and channel dimensions
            x = torch.tensor(iq.astype(np.float32))
            x.unsqueeze_(dim=0)
            x.unsqueeze_(dim=0)

            torch_power = F.energy(x)
            numpy_power = np.mean(np.abs(s) ** 2)

            return numpy_power, torch_power

        for _ in range(n_trials):
            fc = np.random.lognormal()
            a = np.random.lognormal()
            num = np.random.randint(128, 2048 + 1)

            numpy_power, torch_power = _get_numpy_and_torch_energy(fc=fc, a=a, num=num)

            self.assertEqual(torch_power.shape, (1, 1))
            self.assertIsInstance(numpy_power, np.float64)
            self.assertAlmostEqual(numpy_power, torch_power[0, 0].item(), places=2)

    def test_batchoperation(self):
        """Verify that the power calculations hold for independent examples.
        """
        n_trials = 10  # Somewhat expensive because of the batch dimension

        def _get_signal(fc, a, num=1024):
            # Create a complex valued sine wave using numpy
            t = np.linspace(0.0, 100.0, num=num)  # Time
            s = a * np.exp(2 * np.pi * 1j * fc * t)  # Output is complex
            return s

        def _get_torch_iq(s):
            iq = np.stack((s.real, s.imag))
            # Add singleton batch and channel dims
            x = torch.tensor(iq.astype(np.float32))
            x.unsqueeze_(dim=0)
            x.unsqueeze_(dim=0)
            return x

        def _get_power(s, x):
            numpy_power = np.mean(np.abs(s) ** 2)
            torch_power = F.energy(x)
            return numpy_power, torch_power

        for _ in range(n_trials):
            b = np.random.randint(1, 1024)
            c = 1
            num = np.random.randint(128, 2048 + 1)

            # A parallel numpy version isn't being used because its giving
            # incorrect results and its not the version under test anyways so it
            # isn't worth the debugging time to fix it.
            numpy_power_nobatch = np.zeros((b,))

            torch_signal = torch.zeros((b, c, 2, num))
            torch_power_nobatch = torch.zeros((b, 1))

            for i in range(b):
                fc = np.random.lognormal()
                a = np.random.lognormal()

                s = _get_signal(fc=fc, a=a, num=num)
                x = _get_torch_iq(s)

                torch_signal[i, :, :, :] = x

                # Store the value that would have occurred without a batch dim
                n, t = _get_power(s, x)
                numpy_power_nobatch[i] = n
                torch_power_nobatch[i, :] = t

            _, torch_power = _get_power(s, torch_signal)

            self.assertEqual(torch_power.shape, (b, 1))

            for i in range(b):
                # Double verify that the PyTorch implementation agrees with
                # itself when computed in parallel versus one at a time
                self.assertAlmostEqual(
                    torch_power[i, 0].item(), torch_power_nobatch[i, 0].item(), places=2
                )

                # Now that we've verifed that it agrees with itself, then
                # this should almost absolutely succeed (as long as the prior
                # unit test above has succeeded "test_operation")
                self.assertAlmostEqual(
                    numpy_power_nobatch[i], torch_power[i, 0].item(), places=2
                )

    def test_invalidinputs(self):
        """Verify that the function protects itself against bad inputs.
        """
        # No batch/channel dimension
        with self.assertRaises(ValueError):
            F.energy(torch.zeros(2, 1024))

        # No batch dimension
        with self.assertRaises(ValueError):
            F.energy(torch.zeros(1, 2, 1024))

        # Incorrect IQ dimension
        with self.assertRaises(ValueError):
            F.energy(torch.zeros(100, 1, 1, 1024))
