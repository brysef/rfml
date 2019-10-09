"""Plotting helpers to simplify the code flow of Jupyter notebooks.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Iterable, List, Tuple

# Setup a plotting style to cleanup the Jupyter notebooks
# -- This could be annoying to others, but, they'll simply have to set the style after
# importing this file if they want to override it.
sns.set_style("whitegrid")


def plot_IQ(
    iq: np.ndarray, title: str = None, figsize: Tuple[float, float] = (10.0, 5.0)
) -> Figure:
    """Plot IQ data in the time dimension.

    Args:
      iq (np.ndarray): Complex samples in a 2xN numpy array (IQ x Time)
      title (str, optional): Title to put above the plot. Defaults to None.
      figsize (Tuple[float, float], optional): Size of the figure to create.  Defaults
                                               to (10.0, 5.0).

    Raises:
        ValueError: If the IQ array is not 2xN

    Returns:
        [Figure]: Figure that the data was plotted onto (e.g. for saving plot)
    """
    if len(iq.shape) != 2:
        raise ValueError("The IQ array must be complex (e.g. iq.shape=2xN).")
    if iq.shape[0] != 2:
        raise ValueError(
            "The IQ array must be complex (e.g. iq.shape=2xN).  "
            "Your input did not have size 2 in dim 0."
        )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    t = np.arange(iq.shape[1])

    ax.plot(t, iq[0, :], label="Real")
    ax.plot(t, iq[1, :], label="Imag")
    ax.legend()

    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")

    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold")

    return fig


def plot_convergence(
    train_loss: Iterable[float],
    val_loss: Iterable[float],
    title: str = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    annotate: bool = True,
) -> Figure:
    """Plot the convergence of the training/validation loss vs epochs.

    Args:
        train_loss (Iterable[float]): Average training loss for each epoch during
                                      training.
        val_loss (Iterable[float]): Average validation loss for each epoch during
                                    training.
        title (str, optional): Title to put above the plot.  Defaults to None.
        figsize (Tuple[float, float], optional): Size of the figure to create. Defaults
                                                 to (10.0, 5.0).
        annotate (bool, optional): If True, this function will draw lines on the Figure
                                   to mark the best validation loss achieved. Defaults
                                   to True.

    Raises:
        ValueError: If train_loss and val_loss are not the same length
        ValueError: If train_loss and val_loss don't have any data (length is 0)

    Returns:
        Figure: Figure that the convergence was plotted onto (e.g. for saving plot)
    """
    if len(train_loss) != len(val_loss):
        raise ValueError(
            "The loss values for training and validation should have the same length.  "
            "They are of length {} and {} respectively.".format(
                len(train_loss), len(val_loss)
            )
        )
    if len(train_loss) == 0:
        raise ValueError("There must be data to plot (passed lengths were 0).")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    epochs = np.arange(len(train_loss))

    ax.plot(epochs, train_loss, label="Training")
    ax.plot(epochs, val_loss, label="Validation")
    ax.legend()

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold")

    if annotate:
        best_val_loss = np.min(val_loss)
        best_val_epoch = np.argmin(val_loss)
        ax.axhline(best_val_loss, c="k", linestyle="--")
        ax.axvline(best_val_epoch, c="k", linestyle="--")
        ax.text(
            x=best_val_epoch - 0.05,
            y=best_val_loss - 0.02,
            s="Best Validation Loss",
            bbox=dict(facecolor="white", alpha=0.5),
            horizontalalignment="right",
        )

    return fig


def plot_acc_vs_snr(
    acc_vs_snr: Iterable[float],
    snr: Iterable[float],
    title: str = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    annotate: bool = True,
) -> Figure:
    """Plot Classification Accuracy vs Signal-to-Noise Ratio (SNR).

    Args:
        acc_vs_snr (Iterable[float]): Classification accuracy at each SNR.
        snr (Iterable[float]): Signal-to-Noise Ratios (SNR) that were used for
                               evaluation.
        title (str, optional): Title to put above the plot.  Defaults to None.
        figsize (Tuple[float, float], optional): Size of the figure to create. Defaults
                                                  to (10.0, 5.0).
        annotate (bool, optional): If True then the peak accuracy will be annotated with
                                   a horizontal line and with text describing the value.
                                   If False, no lines or text are added on top of the
                                   plotted data.  Defaults to True.

    Raises:
        ValueError: If the lengths of acc_vs_snr and snr do not match

    Returns:
        Figure: Figure that the results were plotted onto (e.g. for saving plot)
    """
    if len(acc_vs_snr) != len(snr):
        raise ValueError(
            "The lengths of acc_vs_snr and snr must match.  "
            "They were {} and {} respectively.".format(len(acc_vs_snr), len(snr))
        )

    # Sort both arrays by SNR to ensure a smoother line plot
    idxs = np.argsort(snr)
    snr = np.array(snr)[idxs]
    acc_vs_snr = np.array(acc_vs_snr)[idxs]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(snr, acc_vs_snr)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Classification Accuracy")

    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold")

    if annotate:
        peak_acc = np.max(acc_vs_snr)
        ax.axhline(peak_acc, c="k", linestyle="--")
        ax.text(
            x=snr[0] + 0.5,
            y=peak_acc - 0.05,
            s="Peak Classification Accuracy ({:.0f}%)".format(peak_acc * 100),
            bbox=dict(facecolor="white", alpha=0.5),
        )

    return fig


def plot_acc_vs_spr(
    acc_vs_spr: Iterable[float],
    spr: Iterable[float],
    title: str = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    annotate: bool = True,
) -> Figure:
    """Plot Classification Accuracy vs Signal-to-Perturbation Ratio (SPR).

    Args:
        acc_vs_spr (Iterable[float]): Classification accuracy at each SPR.
        spr (Iterable[float]): Signal-to-Perturbation Ratios (SPR) that were used for
                               evaluation.
        title (str, optional): Title to put above the plot.  Defaults to None.
        figsize (Tuple[float, float], optional): Size of the figure to create. Defaults
                                                  to (10.0, 5.0).
        annotate (bool, optional): If True then the peak accuracy will be annotated with
                                   a horizontal line and with text describing the value.
                                   If False, no lines or text are added on top of the
                                   plotted data.  Defaults to True.

    Raises:
        ValueError: If the lengths of acc_vs_spr and spr do not match

    Returns:
        Figure: Figure that the results were plotted onto (e.g. for saving plot)
    """
    if len(acc_vs_spr) != len(spr):
        raise ValueError(
            "The lengths of acc_vs_spr and spr must match.  "
            "They were {} and {} respectively.".format(len(acc_vs_spr), len(spr))
        )

    # Sort both arrays by SPR to ensure a smoother line plot
    idxs = np.argsort(spr)
    spr = np.array(spr)[idxs]
    acc_vs_spr = np.array(acc_vs_spr)[idxs]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(spr, acc_vs_spr)

    ax.set_xlabel(r"$E_s/E_p$ (dB)")
    ax.set_ylabel("Classification Accuracy")

    # Left = "least intense attack" -> Right = "most intense attack"
    ax.invert_xaxis()

    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold")

    if annotate:
        peak_acc = np.max(acc_vs_spr)
        ax.axhline(peak_acc, c="k", linestyle="--")
        ax.text(
            x=spr[0] + 0.5,
            y=peak_acc - 0.05,
            s="Peak Classification Accuracy ({:.0f}%)".format(peak_acc * 100),
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

    return fig


def plot_confusion(
    cm: np.ndarray,
    labels: List[str],
    title: str = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
    cmap: Colormap = plt.cm.Blues,
) -> Figure:
    """Plot a confusion matrix.

    Args:
        cm (np.ndarray): NxN array representing the confusion matrix for each
                         true/predicted label pair.
        labels (List[str]): Human readable labels for each classification ID.
        title (str, optional): Title to put above the plot.  Defaults to None.
        figsize (Tuple[float, float], optional): Size of the figure to create. Defaults
                                                  to (10.0, 5.0).
        cmap (Colormap, optional): Colormap to use for the Seaborn Heatmap. Defaults to
                                   plt.cm.Blues.

    Raises:
        ValueError: If the confusion matrix is not square.
        ValueError: If the number of labels doesn't match the confusion matrix shape.

    Returns:
        Figure: Figure that the results were plotted onto (e.g. for saving plot)
    """
    if len(cm.shape) != 2:
        raise ValueError(
            "The confusion matrix must be a square array (NxN), but its shape was {}".format(
                cm.shape
            )
        )
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(
            "The confusion matrix must be a square array (NxN), but its shape was {}".format(
                cm.shape
            )
        )
    if len(labels) != cm.shape[0]:
        raise ValueError(
            "The number of labels provided must match the shape of the confusion "
            "matrix.  You gave {} labels while the confusion matrix had shape "
            "{}".format(len(labels), cm.shape)
        )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    df = pd.DataFrame(cm, index=labels, columns=labels)
    _ = sns.heatmap(df, cmap=cmap, annot=True, square=True, fmt="0.2f", linewidths=1.0)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    if title is not None:
        ax.set_title(title, loc="left", fontweight="bold")

    return fig
