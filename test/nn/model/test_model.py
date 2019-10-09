"""Verify that instantiation and operation of the CNN/CLDNN do not crash.

.. note::

    This doesn't actually check if the model "works" because "works" is pretty
    subjective (what "accuracy" is "good"?).  Therefore, whether it "works" is
    better left to human analysis of experiment outputs but this unit test would
    catch silly mistakes to the architecture such as crashing during operation
    due to incorrectly computing the interior fully connected sizes.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import os
import torch
from unittest import TestCase

# Internal Includes
from rfml.nn.model import CLDNN, CNN


class TestModel(TestCase):
    def _test_construction(self, model_xtor):
        """Verify that the model can be constructed without crashing.
        """
        input_samples = 128
        n_classes = 10
        model = model_xtor(input_samples=input_samples, n_classes=n_classes)

        _ = str(model)
        self.assertEqual(model.n_classes, n_classes)
        self.assertEqual(model.input_samples, input_samples)

    def _test_operation(self, model_xtor):
        """Verify model can be created and provides correct sized outputs.
        """
        n_trials = int(10)

        for _ in range(n_trials):
            input_samples = np.random.randint(128, 1024)
            n_classes = np.random.randint(3, 25)

            model = model_xtor(input_samples=input_samples, n_classes=n_classes)

            b = np.random.randint(128, 512)
            c = 1
            iq = 2

            # The linter doesn't find members of torch
            # pylint: disable=no-member
            x = torch.zeros(b, c, iq, input_samples)
            y = model(x)

            self.assertEqual(y.shape, (b, n_classes))

    def _test_io(self, model_xtor):
        """Verify model can be saved/loaded without crashing.
        """
        model = model_xtor(input_samples=128, n_classes=10)

        # Verify that you can't load the temporary weights before first saving
        with self.assertRaises(RuntimeError):
            model.load()

        # Verify that the temporary weights can be loaded after saving
        model.save()
        model.load()

        # Store the temporary weights path to ensure that it gets cleaned up
        temp_path = model._weights_path
        self.assertTrue(os.path.exists(temp_path))

        # Verify that the "immortal" weights saving works
        immortal_path = "immortal.pt"
        model.save(immortal_path)
        self.assertTrue(os.path.exists(immortal_path))
        model.load(immortal_path)

        # After destruction, the model should cleanup its temporary weights
        del model
        self.assertFalse(os.path.exists(temp_path))

        # And the "immortal" weights are persisted, therefore we have to cleanup
        self.assertTrue(os.path.exists(immortal_path))
        os.remove(immortal_path)

    def _test_freeze(self, model_xtor):
        """Verify that the model tracks its frozen state.

        .. warning::

            Currently this doesn't actually test whether frozen gets implemented
        """
        model = model_xtor(input_samples=128, n_classes=10)

        # Initially the model should not be frozen
        self.assertFalse(model.is_frozen)

        model.freeze()
        self.assertTrue(model.is_frozen)

        model.unfreeze()
        self.assertFalse(model.is_frozen)


class TestCNN(TestModel):
    def test_construction(self):
        self._test_construction(CNN)

    def test_operation(self):
        self._test_operation(CNN)

    def test_io(self):
        self._test_io(CLDNN)

    def test_freeze(self):
        self._test_freeze(CLDNN)


class TestCLDNN(TestModel):
    def test_construction(self):
        self._test_construction(CLDNN)

    def test_operation(self):
        self._test_operation(CLDNN)

    def test_io(self):
        self._test_io(CLDNN)

    def test_freeze(self):
        self._test_freeze(CLDNN)
