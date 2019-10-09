"""Verify that instantiation and operation of the CNN/CLDNN do not crash.

.. note::

    This doesn't actually check if the attack "works" because "works" is pretty
    subjective (what "adversarial success" is "good"?).  Therefore, whether it "works"
    is better left to human analysis of experiment outputs but this unit test would
    catch silly mistakes that cause the code to not at least complete.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import os
import torch
from typing import Tuple
from unittest import TestCase

# Internal Includes
from rfml.attack import fgsm, compute_signed_gradient, scale_perturbation
from rfml.nn.F import energy
from rfml.nn.model import CLDNN, CNN


class TestFGSM(TestCase):
    def _test_fgsm(self, model_xtor, batch_size: int):
        sps = 8
        input_samples = 128
        n_classes = 10
        model = model_xtor(input_samples=input_samples, n_classes=n_classes)

        # Fake up some data to ensure we don't depend on our modem implementations
        x, y = _generate_fake_data(
            input_samples=input_samples, batch_size=batch_size, n_classes=n_classes
        )

        for spr in np.linspace(0.0, 20.0, num=20 + 1):
            _x = fgsm(x=x, y=y, input_size=input_samples, net=model, spr=spr, sps=sps)
            p = x - _x
            perturbation_power = energy(p, sps=sps).detach().numpy()
            # The code base assumes that Es is 1, therefore we can compute spr without
            # a correctly crafted input signal and ensure it still works
            spr_estimate = 10.0 * np.log10(1.0 / np.mean(perturbation_power))

            # Ensure the errror is within 1/10 dB
            self.assertLessEqual(np.abs(spr - spr_estimate), 0.1)

    def _test_compute_signed_gradient(self, model_xtor, batch_size: int):
        input_samples = 128
        n_classes = 10
        model = model_xtor(input_samples=input_samples, n_classes=n_classes)

        # Fake up some data to ensure we don't depend on our modem implementations
        x, y = _generate_fake_data(
            input_samples=input_samples, batch_size=batch_size, n_classes=n_classes
        )

        for spr in np.linspace(0.0, 20.0, num=20 + 1):
            _sg = compute_signed_gradient(x=x, y=y, input_size=input_samples, net=model)

            # Ensure the perturbation size matches the input size (since they're adding)
            for i, (x_shape, sg_shape) in enumerate(zip(x.size(), _sg.size())):
                self.assertEqual(x_shape, sg_shape)

            # Taking absolute value ensures that all of the values should be ~1.0
            _x = torch.abs(_sg)
            _x = _x.detach().numpy()

            # We can then compare the extremes (max/min) to the 1.0 to ensure all match
            self.assertAlmostEqual(np.max(_x), 1.0, places=4)
            self.assertAlmostEqual(np.min(_x), 1.0, places=4)

    def test_scale_perturbation(self):
        sps = 8
        # +/- 1 values
        sg = (np.random.randint(low=0, high=1, size=(100, 1, 2, 128)) * 2) - 1
        sg = torch.from_numpy(sg).float()

        for spr in np.linspace(0.0, 20.0, num=20 + 1):
            p = scale_perturbation(sg=sg, spr=spr, sps=sps)
            perturbation_power = energy(p, sps=sps).detach().numpy()
            # The code base assumes that Es is 1, therefore we can compute spr without
            # a correctly crafted input signal and ensure it still works
            spr_estimate = 10.0 * np.log10(1.0 / np.mean(perturbation_power))

            # Ensure the errror is within 1/10 dB
            self.assertLessEqual(np.abs(spr - spr_estimate), 0.1)

    def test_CNN(self):
        self._test_fgsm(CNN, batch_size=1)
        self._test_fgsm(CNN, batch_size=256)
        self._test_compute_signed_gradient(CNN, batch_size=1)
        self._test_compute_signed_gradient(CNN, batch_size=256)

    def test_CLDNN(self):
        self._test_fgsm(CLDNN, batch_size=1)
        self._test_fgsm(CLDNN, batch_size=256)
        self._test_compute_signed_gradient(CLDNN, batch_size=1)
        self._test_compute_signed_gradient(CLDNN, batch_size=256)


def _generate_fake_data(
    input_samples: int, batch_size: int, n_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, 1, 2, input_samples))
    y = np.random.randint(low=0, high=n_classes, size=batch_size)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x.float(), y.long()
