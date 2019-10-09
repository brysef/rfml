"""Simplistic convolutional neural network.
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch.nn as nn

# Internal Includes
from .base import Model
from rfml.nn.layers import Flatten, PowerNormalization


class CNN(Model):
    """Convolutional Neural Network based on the "VT_CNN2" Architecture

    This network is based off of a network for modulation classification first
    introduced in O'Shea et al and later updated by West/Oshea and Hauser et al
    to have larger filter sizes.

    Modifying the first convolutional layer to not use a bias term is a
    modification made by Bryse Flowers due to the observation of vanishing
    gradients during training when ported to PyTorch (other authors used Keras).

    Including the PowerNormalization inside this network is a simplification
    made by Bryse Flowers so that utilization of DSP blocks in real time for
    data generation does not require knowledge of the normalization used during
    training as that is encapsulated in the network and not in a pre-processing
    stage that must be matched up.

    References
        T. J. O'Shea, J. Corgan, and T. C. Clancy, “Convolutional radio modulation
        recognition networks,” in International Conference on Engineering Applications
        of Neural Networks, pp. 213–226, Springer,2016.

        N. E. West and T. O’Shea, “Deep architectures for modulation recognition,” in
        IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), pp.
        1–6, IEEE, 2017.

        S. C. Hauser, W. C. Headley, and A. J.  Michaels, “Signal detection effects on
        deep neural networks utilizing raw iq for modulation classification,” in
        Military Communications Conference, pp. 121–127, IEEE, 2017.
    """

    def __init__(self, input_samples: int, n_classes: int):
        super().__init__(input_samples, n_classes)

        self.preprocess = PowerNormalization()

        # Batch x 1-channel x IQ x input_samples
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=False,
        )
        self.a1 = nn.ReLU()
        self.n1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=80,
            kernel_size=(2, 7),
            padding=(0, 3),
            bias=True,
        )
        self.a2 = nn.ReLU()
        self.n2 = nn.BatchNorm2d(80)

        # Flatten the input layer down to 1-d
        self.flatten = Flatten()

        # Batch x Features
        self.dense1 = nn.Linear(80 * 1 * input_samples, 256)
        self.a3 = nn.ReLU()
        self.n3 = nn.BatchNorm1d(256)

        self.dense2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.preprocess(x)

        x = self.conv1(x)
        x = self.a1(x)
        x = self.n1(x)

        x = self.conv2(x)
        x = self.a2(x)
        x = self.n2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.a3(x)
        x = self.n3(x)

        x = self.dense2(x)

        return x

    def _freeze(self):
        """Freeze all of the parameters except for the dense layers.
        """
        for name, module in self.named_children():
            if "dense" not in name and "n3" not in name:
                for p in module.parameters():
                    p.requires_grad = False

    def _unfreeze(self):
        """Re-enable training of all parameters in the network.
        """
        for p in self.parameters():
            p.requires_grad = True
