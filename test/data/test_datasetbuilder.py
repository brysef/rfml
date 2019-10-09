"""Verify that the dataset builder works as expected.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
from unittest import TestCase

# Internal Includes
from rfml.data import DatasetBuilder


class TestDatasetBuilder(TestCase):
    def test_Construction(self):
        """Verify that the constructor fails when appropriate.
        """
        # Valid instantiations that should not fail
        _ = DatasetBuilder()
        _ = DatasetBuilder(n=1024)
        _ = DatasetBuilder(keys=["Foo", "Bar"])
        _ = DatasetBuilder(defaults={"Foo": 1, "Bar": 2})
        _ = DatasetBuilder(n=2048, keys=["Foo", "Bar"], defaults={"Foo": 1, "Bar": 2})
        _ = DatasetBuilder(keys=["Foo", "Bar", "Baz"], defaults={"Foo": 1, "Bar": 2})

        # Verify that N must be non-negative
        with self.assertRaises(ValueError):
            _ = DatasetBuilder(n=0)
        with self.assertRaises(ValueError):
            _ = DatasetBuilder(n=-1024)

        # Verify that if keys are defaults are both provided, defaults can't
        # provide additional keys that aren't included in keys
        with self.assertRaises(ValueError):
            _ = DatasetBuilder(
                keys=["Foo", "Bar"], defaults={"Foo": 1, "Bar": 2, "Baz": 3}
            )

    def test_CorrectBuilding(self):
        """Verify that a Dataset is properly created when given valid args.
        """
        db = DatasetBuilder()

        iq = np.zeros((2, 1024))

        db.add(iq, label=1)

        db.add(iq, label=2)
        db.add(iq, label=2)

        db.add(iq, label=3)
        db.add(iq, label=3)
        db.add(iq, label=3)

        db.add(iq, label=4)
        db.add(iq, label=4)
        db.add(iq, label=4)
        db.add(iq, label=4)

        dataset = db.build()

        self.assertEqual(len(dataset), 10)
        self.assertEqual(
            dataset.get_examples_per_class(label="label"), {1: 1, 2: 2, 3: 3, 4: 4}
        )

    def test_IncorrectShape(self):
        """Verify that the assertion on IQ shape works
        """
        # Verify that simply not providing anything close to IQ is failed
        iq = np.zeros(1024)
        db = DatasetBuilder()
        with self.assertRaises(ValueError):
            db.add(iq)

        # Verify that IQ length must match defaults
        iq = np.zeros((2, 1024))
        db = DatasetBuilder(n=128)
        with self.assertRaises(ValueError):
            db.add(iq)

        # Verify that IQ length must match prior rows
        iq = np.zeros((2, 1024))
        db = DatasetBuilder()
        db.add(iq)
        iq = np.zeros((2, 2048))
        with self.assertRaises(ValueError):
            db.add(iq)

    def test_MissingRequiredMeta(self):
        """Verify that the assertion on missing meta data works
        """
        # Verify that the additions must match the constructor provided keys
        iq = np.zeros((2, 1024))
        db = DatasetBuilder(keys=["Foo", "Bar"])
        with self.assertRaises(ValueError):
            db.add(iq)

        # Verify that the defaults can make up for missing keys
        iq = np.zeros((2, 1024))
        db = DatasetBuilder(keys=["Foo", "Bar"], defaults={"Foo": 1})
        db.add(iq, Bar=3)
        db.add(iq, Foo=5, Bar=3)
        with self.assertRaises(ValueError):
            db.add(iq, Foo=2)

        # Verify that the required keys are picked up from prior rows
        iq = np.zeros((2, 1024))
        db = DatasetBuilder()
        db.add(iq, Foo=1, Bar=3)
        with self.assertRaises(ValueError):
            db.add(iq, Foo=2)

    def test_BuildingEmptyDataset(self):
        """Verify that an empty dataset can be created by not calling add
        """
        db = DatasetBuilder()
        dataset = db.build()
        self.assertEqual(len(dataset), 0)
