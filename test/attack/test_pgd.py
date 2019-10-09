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
from rfml.attack import pgd
from rfml.nn.F import energy
from rfml.nn.model import CLDNN, CNN


class TestPGD(TestCase):
    def _test_pgd(self, model_xtor, batch_size: int):
        sps = 8
        input_samples = 128
        n_classes = 10
        model = model_xtor(input_samples=input_samples, n_classes=n_classes)

        # Fake up some data to ensure we don't depend on our modem implementations
        x, y = _generate_fake_data(
            input_samples=input_samples,
            batch_size=batch_size,
            n_classes=n_classes,
            sps=sps,
        )

        for spr in np.linspace(0.0, 20.0, num=20 + 1):
            k = np.random.randint(low=1, high=5)
            _x = pgd(
                x=x, y=y, input_size=input_samples, net=model, spr=spr, sps=sps, k=k
            )
            p = x - _x
            perturbation_power = energy(p, sps=sps).detach().numpy()
            # The code base assumes that Es is 1, therefore we can compute spr without
            # a correctly crafted input signal and ensure it still works
            spr_estimate = 10.0 * np.log10(1.0 / (np.mean(perturbation_power)))

            # Ensure the errror is within 1/10 dB
            self.assertGreaterEqual(spr_estimate, spr)

    def test_CNN(self):
        self._test_pgd(CNN, batch_size=1)
        self._test_pgd(CNN, batch_size=256)

    def test_CLDNN(self):
        self._test_pgd(CLDNN, batch_size=1)
        self._test_pgd(CLDNN, batch_size=256)


def _generate_fake_data(
    input_samples: int, batch_size: int, n_classes: int, sps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = np.random.normal(
        loc=0.0, scale=np.sqrt(1.0 / (2 * sps)), size=(batch_size, 1, 2, input_samples)
    )
    y = np.random.randint(low=0, high=n_classes, size=batch_size)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x.float(), y.long()
