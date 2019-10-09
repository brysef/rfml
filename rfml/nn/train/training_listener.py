"""Interface to receive callbacks about training progress (observer patten).
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"


class TrainingListener(object):
    """Interface for receiving callbacks during training with current progress.

    A user should inherit from this base class, override the corresponding
    callbacks that it intends to receive, and register this listener with a
    trainer in order to receive status updates during training.
    """

    def on_epoch_completed(self, mean_loss: float, epoch: int):
        """Called after a full training epoch has been completed.

        Args:
            mean_loss (float): The mean training loss during this epoch.
            epoch (int): Epoch that was just trained.
        """
        pass

    def on_validation_completed(self, mean_loss: float, epoch: int):
        """Called after the validation loss has been computed.

        Args:
            mean_loss (float): Mean training loss for this epoch.
            epoch (int): Epoch that was just validated.
        """
        pass

    def on_training_completed(
        self, best_loss: float, best_epoch: int, total_epochs: int
    ):
        """Called when a stopping condition has been reached.

        Args:
            best_loss (float): The best loss that was achieved by the model.
            best_epoch (int): The epoch where that loss was achieved.  The
                              weights for the model is reloaded from this epoch.
            total_epochs (int): The total number of epochs that this model
                                trained for.
        """
        pass
