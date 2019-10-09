"""An implementation of adversarial training as described in [Kurakin 2016].

Reference:
    Kurakin, A., Goodfellow, I. J., and Bengio, S. (2016).Adversarial machine learning
    at scale.CoRR, abs/1611.01236.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

# Internal Includes
from .standard import StandardTrainingStrategy
from rfml.attack import fgsm, pgd
import rfml.nn.F as F
from rfml.nn.model import Model


class AdversarialTrainingStrategy(StandardTrainingStrategy):
    """An implementation of adversarial training as described in [Kurakin 2016].

    .. warning::

        This module uses adversarial attacks that scale the input's power when creating
        the perturbation in order to maintain some signal-to-perturbation ratio by
        assuming that the signal as an average energy per symbol of 1.  This isn't a
        problem when this training strategy is used in conjunction with the neural
        networks from this library as the first step in these networks is to scale the
        normalize the input power anyways which undoes any transformation done here.
        However, if, for some reason, your networks do not normalize the input then this
        can lead to undesirable behavior.

    Reference:
        Kurakin, A., Goodfellow, I. J., and Bengio, S. (2016).Adversarial machine
        learning at scale.CoRR, abs/1611.01236.
    """

    def __init__(
        self,
        lr: float = 10e-4,
        max_epochs: int = 50,
        patience: int = 5,
        batch_size: int = 512,
        gpu: bool = True,
        k: float = 0.05,
        n_steps: int = 5,
        spr: float = 10.0,
        adv_method: str = "FGSM",
    ):
        super().__init__(
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            gpu=gpu,
        )
        if k <= 0.0 or k > 1.0:
            raise ValueError(
                "The k value should be in the half open set (0, 1].  You  chose to "
                "specify k as {:0.3f}.  Note -- if you wish to use k=0 then simply use "
                "the standard training strategy.".format(k)
            )
        if n_steps < 1:
            raise ValueError(
                "Must take at least 1 step for the adversarial attack augmentation -- "
                "not {}".format(n_steps)
            )
        if adv_method.upper() not in ["FGSM", "PGD"]:
            raise ValueError(
                "Unkown adversarial methodology to use ({}).  "
                "Supported methods are FGSM and PGD.".format(adv_method)
            )
        self.k = k
        self.n_steps = n_steps
        self.spr = spr

        # The exact value of the sps shouldn't actually matter.  It's simply used for
        # an intermediate scaling of the example before applying the adversarial
        # perturbation with FGSM.  This assumption that it shouldn't matter is based
        # upon the expectation the model does the normalization as the first "layer"
        # in its network.
        self.sps = 8

        self.adv_method = adv_method

    def _train_one_epoch(
        self, model: Model, data: DataLoader, loss_fn: CrossEntropyLoss, optimizer: Adam
    ) -> float:
        total_loss = 0.0
        # Switch the model mode so it remembers gradients, induces dropout, etc.
        model.train()

        for i, batch in enumerate(data):
            x, y = batch

            # Perform adversarial augmentation in the training loop using FGSM
            x = self._adversarial_augmentation(x=x, y=y, model=model)

            # Push data to GPU
            if self.gpu:
                x = Variable(x.cuda())
                y = Variable(y.cuda())
            else:
                x = Variable(x)
                y = Variable(y)

            # Forward pass of prediction -- while some are adversarial
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

    def _adversarial_augmentation(
        self, x: torch.Tensor, y: torch.Tensor, model: Model
    ) -> torch.Tensor:
        # Rely on the fact that the DataLoader shuffles -- therefore can just take the
        # first *n* examples and perform adversarial augmentation on it and it will be
        # a random selection.
        n_adversarial = int(self.k * x.shape[0])
        if n_adversarial == 0:
            return x

        if self.adv_method == "FGSM":
            x[0:n_adversarial, ::] = fgsm(
                x=x[0:n_adversarial, ::],
                y=y[0:n_adversarial],
                net=model,
                spr=self.spr,
                sps=self.sps,
            )
        else:
            x[0:n_adversarial, ::] = pgd(
                x=x[0:n_adversarial, ::],
                y=y[0:n_adversarial],
                net=model,
                k=self.n_steps,
                spr=self.spr,
                sps=self.sps,
            )

        return x
