"""An implementation of a typical multi-class classification training loop.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

# Internal Includes
from .base import TrainingStrategy
from rfml.data import Dataset, Encoder
from rfml.nn.model import Model


class StandardTrainingStrategy(TrainingStrategy):
    """A typical strategy that would be used to train a multi-class classifier.

    Args:
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
    """

    def __init__(
        self,
        lr: float = 10e-4,
        max_epochs: int = 50,
        patience: int = 5,
        batch_size: int = 512,
        gpu: bool = True,
    ):
        super().__init__()
        if lr <= 0.0 or lr >= 1.0:
            raise ValueError(
                "A sane human would choose a learning rate between 0-1, but, you chose "
                "{}".format(lr)
            )
        if max_epochs < 1:
            raise ValueError(
                "You must train for at least 1 epoch, you set the max epochs as "
                "{}".format(max_epochs)
            )
        if patience < 1:
            raise ValueError(
                "Patience must be a positive integer, not {}".format(patience)
            )
        if batch_size < 1:
            raise ValueError(
                "The batch size must be a positive integer, not {}".format(batch_size)
            )

        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.gpu = gpu

    def __call__(
        self, model: Model, training: Dataset, validation: Dataset, le: Encoder
    ):
        """Train the model on the provided training data.

        Args:
            model (Model): Model to fit to the training data.
            training (Dataset): Data used in the training loop.
            validation (Dataset): Data only used for early stopping validation.
            le (Encoder): Mapping from human readable labels to model readable.
        """
        # The PyTorch implementation of CrossEntropyLoss combines the softmax
        # and natural log likelihood loss into one (so none of the models
        # should have softmax included in them)
        criterion = CrossEntropyLoss()

        if self.gpu:
            model.cuda()
            criterion.cuda()

        optimizer = Adam(model.parameters(), lr=self.lr)

        train_data = DataLoader(
            training.as_torch(le=le), shuffle=True, batch_size=self.batch_size
        )
        val_data = DataLoader(
            validation.as_torch(le=le), shuffle=True, batch_size=self.batch_size
        )

        # Fit the data for the maximum number of epochs, bailing out early if
        # the early stopping condition is reached.  Set the initial "best" very
        # high so the first epoch is always an improvement
        best_val_loss = 10e10
        epochs_since_best = 0
        best_epoch = 0
        for epoch in range(0, self.max_epochs):
            train_loss = self._train_one_epoch(
                model=model, data=train_data, loss_fn=criterion, optimizer=optimizer
            )
            self._dispatch_epoch_completed(mean_loss=train_loss, epoch=epoch)

            val_loss = self._validate_once(
                model=model, data=val_data, loss_fn=criterion
            )
            self._dispatch_validation_completed(mean_loss=val_loss, epoch=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_best = 0
                best_epoch = epoch
                model.save()
            else:
                epochs_since_best += 1

            if epochs_since_best >= self.patience:
                break

        # Reload the "best" weights
        model.load()
        self._dispatch_training_completed(
            best_loss=best_val_loss, best_epoch=best_epoch, total_epochs=epoch
        )

    def _train_one_epoch(
        self, model: Model, data: DataLoader, loss_fn: CrossEntropyLoss, optimizer: Adam
    ) -> float:
        total_loss = 0.0
        # Switch the model mode so it remembers gradients, induces dropout, etc.
        model.train()

        for i, batch in enumerate(data):
            x, y = batch

            # Push data to GPU
            if self.gpu:
                x = Variable(x.cuda())
                y = Variable(y.cuda())
            else:
                x = Variable(x)
                y = Variable(y)

            # Forward pass of prediction
            outputs = model(x)

            # Zero out the parameter gradients, because they are cumulative,
            # compute loss, compute gradients (backward), update weights
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / (i + 1.0)
        return mean_loss

    def _validate_once(
        self, model: Model, data: DataLoader, loss_fn: CrossEntropyLoss
    ) -> float:
        total_loss = 0.0
        # Switch the model back to test mode (so that batch norm/dropout doesn't
        # take effect)
        model.eval()
        for i, batch in enumerate(data):
            x, y = batch

            if self.gpu:
                x = x.cuda()
                y = y.cuda()

            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item()

        mean_loss = total_loss / (i + 1.0)
        return mean_loss
