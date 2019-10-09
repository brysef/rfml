"""Verify that the label encoders work as expected.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
from unittest import TestCase

# Internal Includes
from rfml.data import Encoder


_labels = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"]


class TestEncoder(TestCase):
    def test_Operation(self):
        """Verify that encoding/decoding works correctly
        """
        ie = Encoder(labels=_labels, label_name="Not Important")

        for i, v in enumerate(_labels):
            self.assertEqual(ie.encode([v])[0], i)
            self.assertEqual(ie.decode([i])[0], v)

    def test_Attributes(self):
        """Verify that the attributes can be queried
        """
        ie = Encoder(labels=_labels, label_name="Not Important")

        self.assertEqual(ie.labels, _labels)

    def test_NoCrash(self):
        """Verify that the encoder does not crash during operation

        This allows testing the miscellaneous functions that don't necessarily
        require the assertions of input->output relations.
        """
        ie = Encoder(labels=_labels, label_name="Not Important")

        _ = str(ie)
        _ = str(ie.labels)

    def test_LabelNameRetrieval(self):
        """Verify that the encoder stores and reproduces the label_name
        """
        label_str = "Some Column Name"
        ie = Encoder(labels=_labels, label_name=label_str)

        self.assertEqual(ie.label_name, label_str)
