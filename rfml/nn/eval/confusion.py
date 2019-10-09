"""Helper function for computing the confusion matrix of a model on a dataset.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import List, Tuple

# Internal Includes
from rfml.data import Dataset, Encoder
from rfml.nn.model import Model


def compute_confusion(
    model: Model,
    data: Dataset,
    le: Encoder,
    batch_size: int = 512,
    mask: pd.DataFrame.mask = None,
) -> np.ndarray:
    """Compute and normalize a confusion matrix of this model on the dataset.

    Args:
        model (Model): (Trained) model to evaluate.
        data (Dataset): (Testing) data to use for evaluation.
        le (Encoder): Mapping from human readable to machine readable.
        batch_size (int, optional): Defaults to 512.
        mask (pd.DataFrame.mask, optional): Mask to apply to the data before computing
                                            accuracy.  Defaults to None.

    Returns:
        np.ndarray: Normalized Confusion Matrix
    """
    predictions, labels = _extract_predictions_and_labels(
        model=model, data=data, le=le, batch_size=batch_size, mask=mask
    )
    return _confusion_matrix(predictions=predictions, labels=labels, le=le)


def _confusion_matrix(
    predictions: List[int], labels: List[int], le: Encoder
) -> np.ndarray:
    # Note: This could be replaced with sklearn.metrics.confusion_matrix
    # It is simply rewritten in order to avoid introducing a dependency for one thing
    confusion_matrix = np.zeros([len(le.labels), len(le.labels)])

    # Compute the total number of predictions
    for predicted_label, true_label in zip(predictions, labels):
        confusion_matrix[true_label, predicted_label] = (
            confusion_matrix[true_label, predicted_label] + 1.0
        )

    # Normalize these predictions to be a percentage
    for true_label in range(confusion_matrix.shape[0]):
        # Avoid a divide by zero case, no guarantee that we are calling this method
        # with data for all of the classes that could be predicted
        total = np.sum(confusion_matrix[true_label, :])

        if total == 0:
            confusion_matrix[true_label, :] = 0.0
        else:
            confusion_matrix[true_label, :] = confusion_matrix[true_label, :] / total

    return confusion_matrix


def _extract_predictions_and_labels(
    model: Model,
    data: Dataset,
    le: Encoder,
    batch_size: int = 512,
    mask: pd.DataFrame.mask = None,
) -> Tuple[List[int], List[int]]:
    ret_predictions, ret_labels = list(), list()
    dl = DataLoader(
        data.as_torch(le=le, mask=mask), shuffle=True, batch_size=batch_size
    )

    for _, data in enumerate(dl):
        inputs, labels = data
        predictions = model.predict(inputs)
        ret_predictions.extend(predictions)
        ret_labels.extend(labels)

    return ret_predictions, ret_labels
