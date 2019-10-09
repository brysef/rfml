"""Simplistic factory pattern for swapping of datasets.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
from typing import Tuple

# Internal Includes
from .converters import load_RML201610A_dataset, load_RML201610B_dataset
from .dataset import Dataset
from .encoder import Encoder


def build_dataset(
    dataset_name: str, test_pct: float = 0.3, val_pct: float = 0.05, path: str = None
) -> Tuple[Dataset, Dataset, Dataset, Encoder]:
    """Opinionated factory method that allows easy loading of different datasets.

    This method makes an assumption about the labels to use for each dataset -- if you
    need more extensive control then you can call the underlying method directly.

    Args:
        dataset_name (str): Name of the dataset to load.  Currently supported values
                            are:
                            - RML2016.10A
                            - RML2016.10B
        test_pct (float, optional): Percentage of the entire Dataset that should be
                                    withheld as a test set. Defaults to 0.3.
        val_pct (float, optional): Percentage of the non-testing Dataset that should be
                                   split out to use for validation in an early stopping
                                   procedure. Defaults to 0.05.
        path (str, optional): If provided, this is directly passed to the dataset
                              converters so that they do not download the dataset from
                              the internet (a costly operation) if you have already
                              downloaded it yourself. Defaults to None.

    Raises:
        ValueError: If test_pct or val_pct are not between 0 and 1 (non-inclusive).
        ValueError: If the dataset_name is unknown.

    Returns:
        Tuple[Dataset, Dataset, Dataset, Encoder]: train, validation, test, encoder
    """
    if test_pct >= 1.0 or test_pct <= 0.0:
        raise ValueError(
            "Testing percentage must be in the open set between (0-1).  "
            "Not {}.".format(test_pct)
        )
    if val_pct >= 1.0 or val_pct <= 0.0:
        raise ValueError(
            "Validation percentage must be in the open set between (0-1).  "
            "Not {}.".format(test_pct)
        )

    if dataset_name.upper() == "RML2016.10A":
        dataset = load_RML201610A_dataset(path=path)
        le = Encoder(
            [
                "WBFM",
                "AM-DSB",
                "AM-SSB",
                "CPFSK",
                "GFSK",
                "PAM4",
                "BPSK",
                "QPSK",
                "8PSK",
                "QAM16",
                "QAM64",
            ],
            label_name="Modulation",
        )
        on = ["Modulation", "SNR"]
    elif dataset_name.upper() == "RML2016.10B":
        dataset = load_RML201610B_dataset(path=path)
        le = Encoder(
            [
                "WBFM",
                "AM-DSB",
                "CPFSK",
                "GFSK",
                "PAM4",
                "BPSK",
                "QPSK",
                "8PSK",
                "QAM16",
                "QAM64",
            ],
            label_name="Modulation",
        )
        on = ["Modulation", "SNR"]
    else:
        raise ValueError("Unknown dataset ({})".format(dataset_name))

    train, test = dataset.split(frac=test_pct, on=on)
    train, val = dataset.split(frac=val_pct, on=on)

    return train, val, test, le
