"""Helper function for computing the (top-k) accuracy of a model on a dataset.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import pandas as pd
from torch.utils.data import DataLoader
from typing import List, Tuple

# Internal Includes
from rfml.data import Dataset, Encoder
from rfml.nn.model import Model


def compute_accuracy(
    model: Model,
    data: Dataset,
    le: Encoder,
    batch_size: int = 512,
    mask: pd.DataFrame.mask = None,
) -> float:
    """Compute the Top-1 accuracy of this model on the dataset.

    Args:
        model (Model): (Trained) model to evaluate.
        data (Dataset): (Testing) data to use for evaluation.
        le (Encoder): Mapping from human readable to machine readable.
        batch_size (int, optional): Defaults to 512.
        mask (pd.DataFrame.mask, optional): Mask to apply to the data before computing
                                            accuracy.  Defaults to None.

    Returns:
        float: Top-1 Accuracy
    """
    dl = DataLoader(
        data.as_torch(le=le, mask=mask), shuffle=True, batch_size=batch_size
    )

    right = 0
    total = 0

    for _, data in enumerate(dl):
        inputs, labels = data
        predictions = model.predict(inputs)
        right += (predictions == labels).sum().item()
        total += len(labels)

    return float(right) / total


def compute_accuracy_on_cross_sections(
    model: Model, data: Dataset, le: Encoder, column: str, batch_size: int = 512
) -> Tuple[List[float], List]:
    """Compute an accuracy on each unique value in the column (such as SNR or CFO)

    Args:
        model (Model): (Trained) model to evaluate.
        data (Dataset): (Testing) data to use for evaluation.
        le (Encoder): Mapping from human readable to machine readable.
        column (str): Name of the column to use for computing cross sections (e.g. SNR)
        batch_size (int, optional): Defaults to 512.

    Returns:
        List[float], List[object]: Accuracy vs Column, Column Values
    """
    if column not in data.columns:
        raise ValueError(
            "Cannot compute a cross section across the {} column because it does not "
            "exist in the dataset -- {}".format(column, data.columns)
        )

    accuracy = list()
    column_values = list()

    for val in data.df[column].unique():
        mask = data.df[column] == val
        column_values.append(val)
        accuracy.append(
            compute_accuracy(
                model=model, data=data, le=le, mask=mask, batch_size=batch_size
            )
        )

    return accuracy, column_values


def compute_topk_accuracy(
    model: Model,
    data: Dataset,
    le: Encoder,
    k: int,
    batch_size: int = 512,
    mask: pd.DataFrame.mask = None,
) -> float:
    """Computes the probability that the true class is in the top k outputs of the network.

    .. warning::

        If you only want Top-1 Accuracy (if you don't know what this is, then that is what
        you want).  Then you should just use compute_accuracy instead.

    Args:
        model (Model): (Trained) model to evaluate.
        data (Dataset): (Testing) data to use for evaluation.
        le (Encoder): Mapping from human readable to machine readable.
        k (int): Value to use when determining the "top k".
        batch_size (int, optional): Defaults to 512.
        mask (pd.DataFrame.mask, optional): Mask to apply to the data before computing
                                            top-k accuracy.  Defaults to None.

    Returns:
        float: Top-K Accuracy
    """
    dl = DataLoader(
        data.as_torch(le=le, mask=mask), shuffle=True, batch_size=batch_size
    )

    right = 0
    total = 0

    for _, data in enumerate(dl):
        inputs, labels = data
        outputs = model.outputs(inputs)

        # outputs: B x n_classes
        # Therefore, we retrieve the indices of the top k largest values along dim=1
        # ---- note: indices/label integers are interchangable nomenclature
        # topk_predictions: B x k
        topk_predictions = outputs.topk(k=k, dim=1, largest=True).indices

        # Now, we want to retrieve a 0/1 indicator of whether the true label (for that
        # specific batch index) is in the top k labels output by the network
        # labels: B
        # Therefore, we must add a k dimension to the labels and repeat the true label
        # across that dimension -- then we can use element wise equality operators
        # because the sizes will match between topk_predictions and the labels
        labels = labels.view(-1, 1)  # -1 means expand to fit the data
        # labels: B x 1
        labels.expand_as(topk_predictions)
        # Now, the label has been repeated across the second dimension "k times"
        # labels: B x k
        # When computing element wise equality, it is only 1 when the labels match,
        # therefore, the number of correct predictions can simply be a sum of all 1's
        # in the return value from the equality
        right += (topk_predictions == labels).sum().item()
        total += len(labels)

    return float(right) / total
