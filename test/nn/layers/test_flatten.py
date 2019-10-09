"""Verify that the PyTorch flatten layer works.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import torch
from unittest import TestCase

# Internal Includes
from rfml.nn.layers import Flatten


class TestFlatten(TestCase):
    def test_operation(self):
        """Verify that this layer produces correctly sized outputs.
        """
        n_trials = int(10e3)
        flatten = Flatten()

        for _ in range(n_trials):
            b = np.random.randint(128, 1024)
            c = np.random.randint(1, 10)
            iq = 2
            t = np.random.randint(128, 1024)

            x = torch.zeros(b, c, iq, t)
            x = flatten(x)
            self.assertEqual(x.shape, (b, c * iq * t))

    def test_invalidinputs(self):
        """Verify that the layer protects itself against bad inputs.
        """
        flatten = Flatten()

        # No batch feature dimensions
        with self.assertRaises(ValueError):
            flatten(torch.zeros(1024))
