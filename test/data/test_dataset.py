"""Verify that the dataset works as expected.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
from copy import deepcopy
import numpy as np
import os
import torch
from unittest import TestCase

# Internal Includes
from rfml.data import Dataset, DatasetBuilder, Encoder


class TestDataset(TestCase):
    # 5 mods x 5 snrs x 1k examples = 25k entries in the dataset
    MODS = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"]
    SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]
    NEXAMPLES = 1000

    NSAMPLES = 1024

    @classmethod
    def _create_dataset(cls, extras: dict = {}):
        keys = ["SNR", "Modulation"] + list(extras.keys())
        db = DatasetBuilder(n=cls.NSAMPLES, keys=keys)
        iq = np.zeros((2, cls.NSAMPLES))
        for mod in cls.MODS:
            for snr in cls.SNRS:
                for _ in range(cls.NEXAMPLES):
                    db.add(iq, SNR=snr, Modulation=mod, **extras)
        return db.build()

    @classmethod
    def setUpClass(cls):
        """Create a basic dataset for use in the underlying tests.
        """
        cls._data = cls._create_dataset()

    def test_equality(self):
        """Verify that the dataset.__eq__ method works
        """
        db = DatasetBuilder(n=TestDataset.NSAMPLES, keys=["SNR", "Modulation"])
        iq = np.zeros((2, TestDataset.NSAMPLES))
        for i, mod in enumerate(TestDataset.MODS):
            for snr in TestDataset.SNRS:
                for _ in range(i * TestDataset.NEXAMPLES):
                    db.add(iq, SNR=snr, Modulation=mod)
        baddata = db.build()

        copieddata = deepcopy(TestDataset._data)

        # By definition, a dataset must be equivalent to itself
        self.assertEqual(TestDataset._data, TestDataset._data)

        # Verify that equality works in both directions
        self.assertEqual(TestDataset._data, copieddata)
        self.assertEqual(copieddata, TestDataset._data)

        # Verify that inequality works in both directions
        self.assertNotEqual(TestDataset._data, baddata)
        self.assertNotEqual(baddata, TestDataset._data)

    def test_addition(self):
        """Verify that the dataset.__add__ method works
        """
        # Verify a basic example where everything matches
        d1 = TestDataset._create_dataset()
        d2 = TestDataset._create_dataset()
        d3 = TestDataset._create_dataset()
        combined = d1 + d2 + d3

        self.assertEqual(len(combined), len(d1) + len(d2) + len(d3))

        # Verify a harder example, where columns must be dropped
        k1 = {"Extra1": 1, "Extra2": 2}
        d1 = TestDataset._create_dataset(k1)
        k2 = {"Extra2": 2, "Extra3": 3}
        d2 = TestDataset._create_dataset(k2)
        combined = d1 + d2

        self.assertIn("Extra2", combined.df.columns)
        self.assertNotIn("Extra1", combined.df.columns)
        self.assertNotIn("Extra3", combined.df.columns)

    def test_isbalanced(self):
        """Verify that the Dataset can correctly detect imbalance.
        """
        db = DatasetBuilder(n=TestDataset.NSAMPLES, keys=["SNR", "Modulation"])
        iq = np.zeros((2, TestDataset.NSAMPLES))
        for i, mod in enumerate(TestDataset.MODS):
            for snr in TestDataset.SNRS:
                for _ in range(i * TestDataset.NEXAMPLES):
                    db.add(iq, SNR=snr, Modulation=mod)
        baddata = db.build()

        self.assertTrue(TestDataset._data.is_balanced(label="Modulation"))
        self.assertTrue(TestDataset._data.is_balanced(label="SNR"))

        self.assertFalse(baddata.is_balanced(label="Modulation"))
        self.assertTrue(baddata.is_balanced(label="SNR"))

    def test_naivesplit(self):
        """Verify that the Dataset is naively split into two.
        """
        margin = 10e-3  # As percent error
        d1, d2 = TestDataset._data.split(frac=0.3)

        diff1 = len(d1) - 0.7 * len(TestDataset._data)
        diff2 = len(d2) - 0.3 * len(TestDataset._data)
        original = float(len(TestDataset._data))

        self.assertLessEqual(np.abs(diff1 / original), margin)
        self.assertLessEqual(np.abs(diff2 / original), margin)

    def test_intelligentsplit(self):
        """Verify that an intelligent split actually balances classes

        .. note::

            test_isbalanced also ensures that the classes are balanced a priori
            otherwise the test below would fail as well, but, not at the fault
            of the split method
        """
        margin = 10e-3

        d1, d2 = TestDataset._data.split(frac=0.3, on=["SNR", "Modulation"])

        diff1 = len(d1) - 0.7 * len(TestDataset._data)
        diff2 = len(d2) - 0.3 * len(TestDataset._data)
        original = float(len(TestDataset._data))

        self.assertLessEqual(np.abs(diff1 / original), margin)
        self.assertLessEqual(np.abs(diff2 / original), margin)

        self.assertTrue(d1.is_balanced(label="Modulation"))
        self.assertTrue(d1.is_balanced(label="SNR"))

        self.assertTrue(d2.is_balanced(label="Modulation"))
        self.assertTrue(d2.is_balanced(label="SNR"))

    def test_splitmaintainscount(self):
        """Verify a simple split does not change the total number of examples

        This protects against an off-by-one error in the split
        """
        d1, d2 = TestDataset._data.split(frac=0.3, on=["Modulation"])

        current = len(d1) + len(d2)
        original = len(TestDataset._data)

        self.assertEqual(current, original)

    def test_examplesperclass(self):
        """Verify that the examples per class are correctly computed.
        """
        # Verify the modulation examples are computed correctly
        epc = TestDataset._data.get_examples_per_class(label="Modulation")
        self.assertEqual(set(epc.keys()), set(TestDataset.MODS))

        expectation = TestDataset.NEXAMPLES * len(TestDataset.SNRS)
        for actual in epc.values():
            self.assertEqual(actual, expectation)

        # Verify the SNR examples are computed correctly
        epc = TestDataset._data.get_examples_per_class(label="SNR")
        self.assertEqual(set(epc.keys()), set(TestDataset.SNRS))

        expectation = TestDataset.NEXAMPLES * len(TestDataset.MODS)
        for actual in epc.values():
            self.assertEqual(actual, expectation)

    def test_asnumpy(self):
        """Verify that the asnumpy method returns the expected shapes.
        """
        le = Encoder(TestDataset.MODS, label_name="Modulation")
        x, y = TestDataset._data.as_numpy(le=le)

        self.assertEqual(x.shape, (len(TestDataset._data), 1, 2, TestDataset.NSAMPLES))
        self.assertEqual(y.shape, (len(TestDataset._data), 1))

    def test_astorch(self):
        """Verify that the astorch method returns the expected shapes.
        """
        le = Encoder(TestDataset.MODS, label_name="Modulation")
        dataset = TestDataset._data.as_torch(le=le)

        self.assertEqual(len(dataset), len(TestDataset._data))
        x, y = dataset[0]
        self.assertEqual(x.shape, (1, 2, TestDataset.NSAMPLES))
        self.assertEqual(y.dtype, torch.long)
