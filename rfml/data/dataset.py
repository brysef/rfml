"""Wrap a premade dataset inside a Pandas DataFrame.

Provide a wrapper around a Pandas DataFrame for a premade dataset that splits
the classes and other distinguishing factors evenly for training, testing, and
validation sets.  Additionally, this module facilitates data loading from file
and transformation into the format needed by Keras and PyTorch.

By using Pandas masking functionality, this module can be used to subselect
parts of a dataset (e.g. only trained with no frequency offset, a subset of
modulatons, etc.)
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from typing import Dict, List, Tuple
import warnings

# Internal Includes
from .encoder import Encoder


class Dataset(object):
    """Provide a wrapper around a Pandas DataFrame containing a dataset.

    Args:
        df (pd.DataFrame): Pandas DataFrame that represents the dataset
    """

    HDF_KEY = "Dataset"

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        """Directly return the underlying Pandas DataFrame containing the data.

        This can then be used for mask creation.

        Returns:
            pd.DataFrame: Pandas DataFrame that represents the dataset
        """
        # Not providing a "copy" because datasets could be multiple GB and
        # therefore this could become memory prohibitive.  However, allowing
        # access to the internal data structure is also bad defensive
        # programming because a caller could manipulate the underlying data and
        # cause a crash -- For now, assume that the caller is not malicious in
        # order to conserve memory.
        return self._df

    @property
    def columns(self) -> List[str]:
        """Return a list of the columns that are represented in the underlying Dataframe

        Returns:
            List[str]: Column names
        """
        return list(self._df.columns)

    def __len__(self):
        return len(self._df)

    def __eq__(self, other):
        """Compare this Dataset with another to see if they are equivalent

        .. warning::

            This operation is very computationally expensive so ensure that you
            do not try to compare datasets often (such as only restricting this
            to unit tests)
        """
        # Dataframe equality operator doesn't work because of the nested series
        # therefore, it is easiest to just loop over all of the columns and
        # ensure that all elements of each match.
        if not (self._df.columns == other._df.columns).all():
            return False
        # This comparison throws a Value error when there is a mismatch in how
        # it is labeled: "Can only compare identically-labeled Series objects"
        # When this happens, it means they aren't equivalent anyways for our
        # purposes (because they clearly aren't **exactly** equal) and so there
        # is no need to sort the series or drop indexes to get around the error
        try:
            for col in self._df.columns:
                # IQ is stored as a series/array and therefore they require a
                # special floating point comparison (which is very expensive)
                if col in ["I", "Q"]:
                    for r1, r2 in zip(self._df[col], other._df[col]):
                        if not np.allclose(r1, r2):
                            return False
                elif not (self._df[col] == other._df[col]).all():
                    return False
        except ValueError:
            return False
        return True

    def __add__(self, other):
        """Concatenate two Datasets together

        .. warning::

            If either Dataset contains a column that the other does not, this
            function will silently drop that column from the returned Dataset.
        """
        cols1 = self._df.columns
        cols2 = other._df.columns

        to_drop = set(cols1).symmetric_difference(set(cols2))
        df1 = self._df.drop(to_drop, axis=1, errors="ignore")
        df2 = other._df.drop(to_drop, axis=1, errors="ignore")

        combined = pd.concat([df1, df2], sort=False)

        return Dataset(combined)

    def split(
        self, frac: float = 0.3, on: Tuple[str] = None, mask: pd.DataFrame.mask = None
    ) -> Tuple["Dataset", "Dataset"]:
        """Split this Dataset into two based on fractional availability.

        Args:
            frac (float, optional): Percentage of the Dataset to put into the
                second set. Defaults to 0.3.
            on (Tuple[str], optional): Collection of column names, with
                categorical values, to evenly split amongst the two Datasets.
                If provided, each categorical value will have an equal
                percentage representation in the returned Dataset. Defaults to
                None.
            mask (pd.DataFrame.mask, optional): Mask to apply before performing
                the split. Defaults to None.

        Raises:
            ValueError: If *frac* is not between (0, 1)

        Returns:
            Tuple[Dataset, Dataset]: Two Datasets (such as train/validate)

        .. warning::

            Not providing anything for the *on* parameter may lead to incorrect
            behavior.  For instance, you may have a class imbalance in the
            datasets.  This may be desired in some cases, but, its likely one
            would want to explicitly specify this and not rely on randomness.

        .. seealso:: Dataset.subsample
        """
        if frac <= 0.0 or frac >= 1.0:
            raise ValueError("frac must be between (0, 1), not {}".format(frac))

        df = self._df

        # Mask out data if necessary
        if mask is not None:
            df = df[mask]

        # Shuffle the dataset
        df = df.sample(frac=1)

        if on is None or len(on) == 0:
            # No need to preserve any notion of "evenness" so can directly split
            # the DataFrame here
            idx = int(len(df) * frac) + 1
            df1 = df[idx:]
            df2 = df[:idx]
            return Dataset(df1), Dataset(df2)

        # Use a private function for the ability to use recursion
        def _splitDF(subDF, frac, on):
            col = on[0]
            if len(on) > 1:
                on = on[1:]
            else:
                on = None

            ret0List = list()
            ret1List = list()

            for val in subDF[col].unique():
                _subDF = subDF[subDF[col] == val].copy()

                if on is None:
                    idx = int(len(_subDF) * frac) + 1
                    ret0 = _subDF[idx:]
                    ret1 = _subDF[:idx]
                else:
                    ret0, ret1 = _splitDF(_subDF, frac, on)

                ret0List.append(ret0)
                ret1List.append(ret1)

            return pd.concat(ret0List), pd.concat(ret1List)

        df1, df2 = _splitDF(df, frac, on)

        return Dataset(df1), Dataset(df2)

    def as_numpy(
        self, le: Encoder, mask: pd.DataFrame.mask = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode the Dataset as a machine learning <X, Y> pair in NumPy format.

        Args:
            le (Encoder): Label encoder used to translate the label column into
                a format the neural network will understand (such as an index).  The
                label column is embedded within this class.
            mask (pd.DataFrame.mask, optional): Mask to apply before creating
                the Machine Learning pairs. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: x, y

        The X matrix is returned in the format (batch, channel, iq, time).
        The Y matrix is returned in the format (batch).

        Batch corresponds to the number of examples in the dataset, channel is
        always 1, IQ is always 2, and time is variable length depending on how
        the underlying data has been sliced.

        .. note::

            Numpy is the format used by Keras.  Other machine learning
            frameworks (such as PyTorch) require a separate method for getting
            the data ready.

        .. seealso:: rfml.data.Encoder,
                     rfml.data.Dataset.as_torch
        """
        features = ["I", "Q"]
        df = self._df

        # Mask out data if necessary
        if mask is not None:
            df = df[mask]

        x = np.array(df[features].values.tolist())
        # Add in the channel dimension
        x = np.reshape(x, [x.shape[0], 1, x.shape[1], x.shape[2]])
        y = np.vstack(le.encode(df[le.label_name]))

        return x, y

    def as_torch(self, le: Encoder, mask: pd.DataFrame.mask = None) -> TensorDataset:
        """Encode the Dataset as machine learning <X, Y> pairs in PyTorch
        format.

        Args:
            le (Encoder): Label encoder used to translate the label column into
                a format the neural network will understand (such as an index).  The
                label column is embedded within this class.
            mask (pd.DataFrame.mask, optional): Mask to apply before creating
                the Machine Learning pairs. Defaults to None.

        Returns:
            TensorDataset: Dataset to be used in training or testing loops.

        The X matrix is returned in the format (batch, channel, iq, time).
        The Y matrix is returned in the format (batch).

        Batch corresponds to the number of examples in the dataset, channel is
        always 1, IQ is always 2, and time is variable length depending on how
        the underlying data has been sliced.

        .. note::

            TensorDataset is the format used by PyTorch and allows for iteration
            in batches.  For other machine learning frameworks, such as Keras,
            ensure you call the correct method.

        .. seealso:: rfml.data.Encoder,
                     rfml.data.Dataset.as_numpy
        """
        x, y = self.as_numpy(le=le, mask=mask)

        # Ensure that the labels look like an array
        y = y.squeeze(axis=1)

        # Convert to the correct data types for PyTorch
        x = x.astype(np.float32)
        y = y.astype("int64")

        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

        return dataset

    def get_examples_per_class(self, label: str = "Modulation") -> Dict[str, int]:
        """Count the number of examples per class in this Dataset.

        Args:
            label (str, optional): Column that is used as the class label.
                Defaults to "Modulation".

        Returns:
            Dict[str, int]: Count of examples (value) per label (key).
        """
        counts = self._df[label].value_counts()
        return counts.to_dict()

    def is_balanced(self, label: str = "Modulation", margin: int = 0) -> bool:
        """Check if the data contained in this dataset is evenly represented by
        a categorical label.

        Args:
            label (str, optional): The column of the data to verify is balanced.
                Defaults to "Modulation".
            margin (int, optional): Difference between the expected balance and
                the true balance before this check would fail.  This can be
                useful for checking for a "fuzzy balance" that would occur if
                the Dataset was previously split and therefore the length of the
                Dataset is no longer divisible by the number of categorical
                labels. Defaults to 0.

        Returns:
            bool: True if the Dataset is balanced, False otherwise.
        """
        n = len(self)
        epc = self.get_examples_per_class(label=label)
        for c in epc.values():
            diff = c - n / len(epc.keys())
            if np.abs(diff) > margin:
                return False
        return True
