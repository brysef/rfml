"""Simplistic factory pattern for swapping of training strategies.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# Internal Includes
from .adversarial import AdversarialTrainingStrategy
from .base import TrainingStrategy
from .standard import StandardTrainingStrategy


def build_trainer(
    strategy: str,
    lr: float = 10e-4,
    max_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 512,
    gpu: bool = True,
    **kwargs
) -> TrainingStrategy:
    """Construct a training strategy from the given parameters.

    Args:
        strategy (str): Strategy to use when training the network, current
                        options are:
                            - "standard"
        lr (float, optional): Learning rate to be used by the optimizer.
                              Defaults to 10e-4.
        max_epochs (int, optional): Maximum number of epochs to train before
                                    stopping training to preserve computing
                                    resources (even if the network is still
                                    improving). Defaults to 50.
        patience (int, optional): Maximum number of epochs to continue to train
                                  for even if the network is not still
                                  improving before deciding that overfitting is
                                  occurring and stopping. Defaults to 5.
        batch_size (int, optional): Number of examples to give to the model at
                                    one time.  If this value is set too high,
                                    then an out of memory error could occur.  If
                                    the value is set too low then training will
                                    take a longer time (and be more variable).
                                    Defaults to 512.
        gpu (bool, optional): Flag describing whether the GPU is used or the
                              training is performed wholly on the CPU.
                              Defaults to True.
        **kwargs: The remainder of the keyword arguments are directly passed through to
                the constructor of the class being instantied.

    Example:
        >>> trainer = build_trainer("standard")
        >>> trainer(model, training, validation, encoder)
        >>> model.save("/path/to/weights.pt")
    """
    if strategy.upper() == "STANDARD":
        return StandardTrainingStrategy(
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            gpu=gpu,
        )
    elif strategy.upper() == "ADVERSARIAL":
        return AdversarialTrainingStrategy(
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            gpu=gpu,
            **kwargs
        )
    else:
        raise ValueError("Unknown training strategy ({})".format(strategy))
