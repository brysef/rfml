"""Simple helper class for encoding/decoding the labels for classification

.. note::

    While many packages like sklearn and keras provide similar functionality,
    they were all quite annoying and did not play well with others.  Since this
    functionality is so simple, its easier to just write our own implementation.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
from copy import deepcopy
import numpy as np
import torch
from typing import Dict, Tuple


class Encoder(object):
    """Encode the labels as an index of the "one-hot" which is used by PyTorch.

    Args:
        labels (Tuple[str]): A collection of human readable labels that could be
            encountered
        label_name (str): Name of the label column in the dataset that is being
            categorically encoded.

    Examples:

        >>> "WBFM" -> 1
        >>> "QAM16" -> 6
    """

    def __init__(self, labels: Tuple[str], label_name: str):
        self._labels = labels

        self._encoder = dict()
        self._decoder = dict()

        for i, label in enumerate(labels):
            encoded = i
            self._encoder[label] = encoded
            self._decoder[encoded] = label

        self._label_name = label_name

    @property
    def labels(self) -> Tuple[str]:
        """A collection of human readable labels that could be encountered --
        This allows the extraction of these labels by another object in order
        to plot or log.
        """
        return deepcopy(self._labels)

    @property
    def label_name(self) -> str:
        """The name of the column in the dataset that is categorically encoded by this
        class.
        """
        return self._label_name

    def __repr__(self) -> str:
        return "Encoder({}, {})".format(self._labels, self._label_name)

    def __str__(self) -> str:
        ret = ""
        for k, v in self._encoder.items():
            ret = ret + "{k}: {v}\n".format(k=k, v=v)
        return ret

    def __len__(self) -> int:
        return len(self._labels)

    def encode(self, labels: Tuple[str]) -> Tuple[int]:
        """Encode a list of human readable labels into machine readable labels.

        Args:
            labels (Tuple[str]): Human readable labels to encode.

        Returns:
            Tuple[int]: A collection of machine readable labels.
        """
        ret = list()
        for label in labels:
            ret.append(self._encoder[label])
        return ret

    def decode(self, encoding: Tuple[int]) -> Tuple[str]:
        """Decode a list of machine readable labels into human readable labels.

        Args:
            encoding (Tuple[int]): A collection of machine readable labels.

        Returns:
            Tuple[str]: A collection of human readable labels.
        """
        ret = list()
        for le in encoding:
            ret.append(self._decoder[le])
        return ret
