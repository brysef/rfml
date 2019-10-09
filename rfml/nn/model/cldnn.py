"""Convolutional long deep neural network (CNN + GRU + MLP)
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch
import torch.nn as nn

# Internal Includes
from .base import Model
from rfml.nn.layers import Flatten, PowerNormalization


class CLDNN(Model):
    """Convolutional Long Deep Neural Network (CNN + GRU + MLP)

    This network is based off of a network for modulation classification first
    introduced in West/O'Shea.

    The following modifications/interpretations were made by Bryse Flowers:

    - Batch Normalization was added otherwise the model was not stable enough to train
      in many cases (its unclear whether this was included in West's model)
    - The filter sizes were changed to 7 and the padding was set to 3 (where as
      West's paper said they used size 8 filters and did not mention padding)
        - An odd sized filter is necessary to ensure that the intermediate
          signal/feature map lengths are the same size and thus can be concatenated
          back together
    - A Gated Recurrent Unit (GRU) was used in place of a Long-Short Term Memory (LSTM).
        - These two submodules should behave nearly identically but GRU has one fewer
          equation
    - Bias was not used in the first convolution in order to more closely mimic the
      implementation of the CNN.
    - The hidden size of the GRU was set to be the number of classes it is trying to
      predict -- it makes the most sense instead of trying to find an arbritrary best
      hidden size.

    References
        N. E. West and T. O’Shea, “Deep architectures for modulation recognition,” in
        IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), pp.
        1–6, IEEE, 2017.
    """

    def __init__(self, input_samples: int, n_classes: int):
        super().__init__(input_samples, n_classes)

        # Batch x 1-channel x IQ x input_samples
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=50,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=False,
        )
        self.a1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(50)

        self.conv2 = nn.Conv2d(
            in_channels=50,
            out_channels=50,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=True,
        )
        self.a2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(50)

        self.conv3 = nn.Conv2d(
            in_channels=50,
            out_channels=50,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=True,
        )
        self.a3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(50)

        # Flatten along channels and I/Q
        self.flatten_preserve_time = Flatten(preserve_time=True)

        self.GRU_n_layers = 1
        self.GRU_n_directions = 1
        self.GRU_hidden_size = n_classes
        self.gru = nn.GRU(
            input_size=100 * 2,  # 100 channels after concatenation (50+50) * IQ (2)
            hidden_size=self.GRU_hidden_size,
            batch_first=True,
            num_layers=self.GRU_n_layers,
            bidirectional=False,
        )

        # Flatten everything outside of batch dimension
        self.flatten = Flatten()

        # Fully connected layers
        # All of the outputs of the GRU are taken (instead of just the final hidden
        # output after all of the time samples).  Therefore, the number of "features"
        # after flattening is the time length * the hidden size * number of directions
        self.dense1 = nn.Linear(
            input_samples * self.GRU_hidden_size * self.GRU_n_directions, 256
        )
        self.a4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(256)

        self.dense2 = nn.Linear(256, n_classes)

    def forward(self, x):
        channel_dim = 1
        batch_size = x.shape[0]

        # Up front "filter" with no bias
        x = self.conv1(x)
        x = self.a1(x)
        a = self.bn1(x)  # Output is concatenated back as a "skip connection" below

        # Convolutional feature extraction layers
        x = self.conv2(a)
        x = self.a2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.a3(x)
        x = self.bn3(x)

        # Concatenate the "skip connection" with the output of the rest of the CNN
        # pylint: disable=no-member
        x = torch.cat((a, x), dim=channel_dim)

        # Temporal feature extraction
        x = self.flatten_preserve_time(x)  # BxTxF
        hidden = x.new(
            self.GRU_n_layers * self.GRU_n_directions, batch_size, self.GRU_hidden_size
        )
        hidden.zero_()
        x, _ = self.gru(x, hidden)

        # MLP Classification stage
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.a4(x)
        x = self.bn4(x)

        x = self.dense2(x)

        return x

    def _freeze(self):
        """Freeze all of the parameters except for the dense layers.
        """
        for name, module in self.named_children():
            if "dense" not in name and "bn4" not in name:
                for p in module.parameters():
                    p.requires_grad = False

    def _unfreeze(self):
        """Re-enable training of all parameters in the network.
        """
        for p in self.parameters():
            p.requires_grad = True
