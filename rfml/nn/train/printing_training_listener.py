"""TrainingListener for quick prototypes that only log to stdout.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# Internal Includes
from .training_listener import TrainingListener


class PrintingTrainingListener(TrainingListener):
    """TrainingListener implementation for quick prototypes that only logs to stdout.
    """

    def on_epoch_completed(self, mean_loss: float, epoch: int):
        print("Epoch {epoch} completed!".format(epoch=epoch))
        print("\t\t-Mean Training Loss: {mean_loss:.3f}".format(mean_loss=mean_loss))

    def on_validation_completed(self, mean_loss: float, epoch: int):
        print("\t\t-Mean Validation Loss: {mean_loss:.3f}".format(mean_loss=mean_loss))

    def on_training_completed(
        self, best_loss: float, best_epoch: int, total_epochs: int
    ):
        print("Training has Completed:\n")
        print("=======================")
        print("\tBest Validation Loss: {best_loss:.3f}".format(best_loss=best_loss))
        print("\tBest Epoch: {best_epoch:d}".format(best_epoch=best_epoch))
        print("\tTotal Epochs: {total_epochs:d}".format(total_epochs=total_epochs))
        print("=======================")
