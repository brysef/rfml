"""Provide a builder pattern for the creation of a dataset.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import pandas as pd
from typing import Dict, Set, Union

# Internal Includes
from .dataset import Dataset


class DatasetBuilder(object):
    """Builder pattern for programmatic creation of a Dataset

    Args:
        n (int, optional): Length of the time window (number of samples)
            that each entry in the dataset should have.  If it is not
            provided, then it is inferred from the first added example.
            Defaults to None.
        keys (Set[str], optional): A set of column headers that will
            be included as metadata for all examples.  If it is not
            provided, then it is inferred from the first added example.
            Subsequent examples that are added must either have all of
            these keys provided as metadata or they must be defined in
            the defaults below. Defaults to None.
        defaults (Dict[str, Union, optional): A mapping of default
            metadata values that will be included for each example if
            they aren't overridden. Defaults to dict().

    Examples:

        >>> iq = np.zeros((2, 1024))
        >>> db = DatasetBuilder()
        >>> db.add(iq, Modulation="BPSK")
        >>> db.add(iq, Modulation="QPSK")
        >>> dataset = db.build()

    Raises:
        ValueError: If both keys and defaults are provided, but, the
            defaults have additional keys that were not provided.
        ValueError: If n is negative or 0.

    .. seealso:: rfml.data.Dataset
    """

    def __init__(
        self,
        n: int = None,
        keys: Set[str] = None,
        defaults: Dict[str, Union[str, int, float]] = dict(),
    ):
        if n is not None and n <= 0:
            raise ValueError(
                "The number of time samples for the Dataset must be non-negative "
                "({})".format(n)
            )
        if (
            keys is not None
            and defaults is not None
            and not set(defaults.keys()).issubset(set(keys))
        ):
            raise ValueError(
                "The keys of the defaults must be a subset of the required keys "
                "provided."
            )

        self._n = n
        self._keys = None
        if keys is not None:
            # Allows a list input by silently converting it to a set
            self._keys = set(keys).union(set(["I", "Q"]))
        self._defaults = defaults

        self._rows = list()

    def add(self, iq: np.ndarray, **kwargs) -> "DatasetBuilder":
        """Add a new example to the Dataset that is being built.

        Args:
            iq (np.ndarray): A (2xN) array of IQ samples.
            **kwargs: Each key=value pair is included as metadata for this
                example.

        Returns:
            DatasetBuilder: By returning the self, these calls can be chained.

        Raises:
            ValueError: If the IQ data does not match the expected shape -- It
                should be (2xN) where N has been provided during construction of
                this builder or inferred from the first example added.
            ValueError: If all of the necessary metadata values are not provided
                in kwargs.  The necessary metadata values are either provided
                during construction of this builder or inferred from the first
                example added.
        """
        # Verify that the IQ dimensions are valid
        if len(iq.shape) != 2:
            raise ValueError(
                "The IQ array must be two dimensional, not {} dimensions.".format(
                    len(iq.shape)
                )
            )
        if self._n is None:
            self._n = iq.shape[1]
        elif iq.shape[1] != self._n:
            raise ValueError(
                "The IQ array must contain {} samples, not {} samples.".format(
                    self._n, iq.shape[1]
                )
            )

        # Construct the desired keys if they haven't already been
        if self._keys is None:
            self._keys = set(kwargs.keys())
            self._keys = self._keys.union(set(["I", "Q"]))
            self._keys = self._keys.union(set(self._defaults.keys()))

        # Construct the actual row candidate
        row = dict()

        row["I"] = iq[0, :]
        row["Q"] = iq[1, :]

        row.update(self._defaults)
        row.update(kwargs)

        # Verify that all of the necessary keys, and only those, are going to be
        # added into the dataset.
        keys = set(row.keys())
        missing = self._keys - keys
        extras = keys - self._keys

        if len(missing) != 0:
            raise ValueError(
                "The added example is missing {} keys from the metadata that should be "
                "provided ({}).".format(len(missing), missing)
            )
        if len(extras) != 0:
            raise ValueError(
                "The added example has {} additional keys in the metadata that the "
                "other examples did not ({}).".format(len(extras), extras)
            )

        self._rows.append(row)

        return self

    def build(self) -> Dataset:
        """Build the Dataset based on the examples that have been added.

        Returns:
            Dataset: A compiled dataset consisting of the added examples.
        """
        df = pd.DataFrame(self._rows, columns=self._keys)
        return Dataset(df)
