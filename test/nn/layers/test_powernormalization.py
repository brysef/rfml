"""Verify that the PyTorch power normalization works.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
from itertools import product
import numpy as np
import torch
from unittest import TestCase

# Internal Includes
from rfml.nn.layers import PowerNormalization


class TestPowerNormalization(TestCase):
    def test_operation(self):
        """Verify that all signals going through this layer have unit power.
        """
        n_trials = int(10)  # Somewhat expensive because of the batch dimension
        normalize = PowerNormalization()

        def _get_input(num):
            # Generate random parameters for the signal
            fc = np.random.lognormal()
            a = np.random.lognormal()

            # Create a complex valued sine wave using numpy
            t = np.linspace(0.0, 100.0, num=num)  # Time
            s = a * np.exp(2 * np.pi * 1j * fc * t)  # Output is complex
            iq = np.stack((s.real, s.imag))

            # Convert the numpy array to a tensor, adding in batch/channel
            x = torch.tensor(iq.astype(np.float32))
            x.unsqueeze_(dim=0)
            x.unsqueeze_(dim=0)

            return x

        def _get_batched_inputs():
            # Create a random number of examples int he batch dimension
            # and a random length signal input -- it should work over all
            b = np.random.randint(128, 1024 + 1)
            num = np.random.randint(128, 2048 + 1)

            x = torch.zeros(b, 1, 2, num)
            for i in range(b):
                x[i, :, :, :] = _get_input(num=num)
            return x

        def _get_power(x):
            # Convert from PyTorch back to a complex valued array
            # B x C x IQ x T
            x = x.numpy()

            # Calculate the power of each channel of each example
            power = np.zeros((x.shape[0], x.shape[1]))
            for b, c in product(range(x.shape[0]), range(x.shape[1])):
                iq = x[b, c, :, :]
                s = iq[0, :] + 1j * iq[1, :]
                power[b, c] = np.mean(np.abs(s) ** 2)

            return power

        for _ in range(n_trials):
            x = _get_batched_inputs()
            before = x.shape

            x = normalize(x)
            after = x.shape

            # Verify the normalization operation didn't change the shape
            self.assertEqual(before, after)

            # Verify that all examples have unit power
            power = _get_power(x)
            for b, c in product(range(power.shape[0]), range(power.shape[1])):
                self.assertAlmostEqual(power[b, c], 1.0, places=3)

    def test_invalidinputs(self):
        """Verify that the layer protects itself against bad inputs.
        """
        normalize = PowerNormalization()

        # No batch/channel dimension
        with self.assertRaises(ValueError):
            normalize(torch.zeros(2, 1024))

        # No batch dimension
        with self.assertRaises(ValueError):
            normalize(torch.zeros(1, 2, 1024))

        # Incorrect IQ dimension
        with self.assertRaises(ValueError):
            normalize(torch.zeros(100, 1, 1, 1024))
