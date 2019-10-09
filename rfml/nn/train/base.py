"""Base class that all trainers inherit from (strategy pattern).
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# Internal Includes
from .training_listener import TrainingListener


class TrainingStrategy(object):
    def __init__(self):
        self._listeners = list()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "The base TrainingStrategy should not be "
            "called.  Make sure you are instantiating "
            "a child class."
        )

    def register_listener(self, listener: TrainingListener):
        """Register a callback to receive events about training progress.

        Args:
            listener (TrainingListener): Listener that will be called on each
                                         status update.  The user should
                                         override the methods for the
                                         corresponding events it desires to
                                         receive.

        .. warning::

            Currently, this class trusts the listeners not to crash and does
            not catch their exceptions thrown in order to ensure that any errors
            are passed all the way up the stack.

        .. seealso:: rfml.nn.train.TrainingListener
        """
        self._listeners.append(listener)

    def _dispatch_stage_starting(self, stage: str):
        for listener in self._listeners:
            listener.on_stage_starting(stage=stage)

    def _dispatch_epoch_completed(self, mean_loss: float, epoch: int):
        for listener in self._listeners:
            listener.on_epoch_completed(mean_loss=mean_loss, epoch=epoch)

    def _dispatch_validation_completed(self, mean_loss: float, epoch: int):
        for listener in self._listeners:
            listener.on_validation_completed(mean_loss=mean_loss, epoch=epoch)

    def _dispatch_training_completed(
        self, best_loss: float, best_epoch: int, total_epochs: int
    ):
        for listener in self._listeners:
            listener.on_training_completed(
                best_loss=best_loss, best_epoch=best_epoch, total_epochs=total_epochs
            )
