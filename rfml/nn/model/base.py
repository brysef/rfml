"""Base class for all neural network models
"""
__author__ = "Bryse Flowers <brysef@vt.edu>"

# External Includes
import torch
from torch.autograd import Variable
import os
from os.path import exists
from uuid import uuid4


class Model(torch.nn.Module):
    """Base class that all neural network models inherit from.

    Args:
        input_samples (int): The number of samples that will be given to this
                             Model for each inference.
        n_classes (int): The number of classes that this Model will predict.

    This model supports standard switching between training/evaluation through
    PyTorch (e.g. Model.train() and Model.eval()) but also supports a custom
    command to allow transfer learning by freezing only portions of the network
    (e.g. Model.freeze() and Model.unfreeze()).  Note that some subclasses of
    this may not necessarily support this feature.

    This class also provides all of the common functionality to the child
    classes such as save() and load().
    """

    # pylint: disable=no-member
    # The linter doesn't correctly find the members of the torch library

    def __init__(self, input_samples: int, n_classes: int):
        super().__init__()
        self._input_samples = input_samples
        self._n_classes = n_classes
        self._frozen = False

    def __del__(self):
        # Delete the temporary weights if they have been used and still exist
        if hasattr(self, "_weights_path") and exists(self._weights_path):
            os.remove(self._weights_path)

    def __str__(self):
        ret = super().__str__() + "\n"
        ret += "----------------------\n"
        ret += "Trainable Parameters: {}\n".format(self.n_trainable_parameters)
        ret += "Fixed Parameters: {}\n".format(
            self.n_parameters - self.n_trainable_parameters
        )
        ret += "Total Parameters: {}\n".format(self.n_parameters)
        ret += "----------------------\n"
        return ret

    @property
    def device(self) -> torch.device:
        """Retrieve the most probable device that this model is currently on.

        .. warning::

            This method is not guaranteed to work if the model is split onto multiple
            devices (e.g. part on CPU, part on GPU 1, and part on GPU 2).

        Returns:
            torch.device: Device that this model is likely located on
        """
        probable_device = next(self.parameters()).device
        return probable_device

    def freeze(self):
        """Freeze part of the model so that only parts of the model are updated.
        """
        self._frozen = True
        # Call the child classes implementation
        if hasattr(self, "_freeze"):
            self._freeze()

    def unfreeze(self):
        """Re-enable learning of all parts of the model.
        """
        self._frozen = False
        # Call the child classes implementation
        if hasattr(self, "_unfreeze"):
            self._unfreeze()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return a categorical prediction using an argmax strategy.

        Args:
            x (torch.Tensor): Inputs to the network.

        Returns:
            torch.Tensor: Label of the highest class for each input.

        .. seealso:: outputs
        """
        self.eval()
        x = Variable(x.to(self.device))

        y = self(x)
        # batch x n_classes -- therefore, take the max along the classes dim
        predictions = torch.argmax(y, dim=1)

        return predictions.cpu()

    def outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method for receiving the full outputs of the neural network.

        .. note::

            This method should only be used during testing -- training should operate
            directly on the forward/backward calls provided by PyTorch.

        This method is opinionated in order to reduce complexity of receiving model
        outputs for the caller.  To that end, it does four things for the caller:

            - Puts the model in *eval* mode so that Batch Normalization/Dropout aren't
              induced
            - Pushes the data to whatever device this model is currently on (such as
              cuda:0/cuda:1/cpu/etc.) so the caller doesn't have to know where the model
              resides
            - Obtains the outputs of the network (using whichever device the model is
              currently on)
            - Pulls the outputs back to CPU for further analysis by the caller

        Args:
            x (torch.Tensor): Inputs to the network.

        Returns:
            torch.Tensor: Full outputs of this network for each input.

        .. seealso: predict
        """
        self.eval()
        x = Variable(x.to(self.device))

        y = self(x)

        return y.cpu()

    @property
    def is_frozen(self):
        return self._frozen

    @property
    def n_trainable_parameters(self):
        """The number of parameters that would be 'learned' during training.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def n_parameters(self):
        """The total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def input_samples(self):
        """The expected number of complex samples on the input to this model.
        """
        return self._input_samples

    @property
    def n_classes(self):
        """The number of outputs of this model per inference.
        """
        return self._n_classes

    def load(self, path: str = None, map_location: str = "cpu"):
        """Load pretrained weights from disk.

        Args:
            path (str, optional): If provided, then load immortal weights from
                                  this path.  If not set, then the temporary
                                  weights path is used (for reloading the
                                  "best weights" in an early stopping
                                  procedure). Defaults to None.
            map_location (str, optional): String representing the device to
                                          load the model/weights into. If this
                                          is set to None, then the weights will
                                          be loaded onto the same device they
                                          were saved from. This can cause
                                          failures if the devices do not exist
                                          on the machine calling this function.
                                          This can occur if the model is
                                          trained on one device (with GPUs) and
                                          then used on another device where
                                          GPUs do not exist.  It can also occur
                                          on the same device if the GPU
                                          configurations are changed (by
                                          setting CUDA_VISIBLE_DEVICES) or if
                                          the desired device is out of memory.
                                          See torch.load() documentation for
                                          further details and options as this
                                          parameter is simply a passthrough for
                                          that. Defaults to "cpu" if path is
                                          provided, else it is set to None and
                                          the input provided by the user is
                                          ignored.

        .. warning::

            This doesn't provide safety against weights paths existing;
            therefore, it will throw the arguments back up the stack instead
            of silencing them.
        """
        if path is None:
            if hasattr(self, "_weights_path"):
                path = self._weights_path
                map_location = None
            else:
                raise RuntimeError(
                    "Told to load a model without a path to pretrained weights when "
                    "this model has never been saved to a temporary location either."
                )

        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint)

    def save(self, path: str = None):
        """Save the currently loaded/trained weights to disk.

        Args:
            path (str, optional): If provided, the weights will be saved at this
                                  path, which is useful for immortalizing the
                                  weights once training is completed.  If not
                                  provided, then the model will create a
                                  temporary file with a unique ID to store
                                  the current weights at, and delete that file
                                  when this object is deleted.  This can be
                                  useful for storing intermediate weights
                                  that will be used to reload "the best weights"
                                  for an early stopping procedure, without
                                  requiring the caller to care where they are
                                  stored at.  Defaults to None.

        .. warning::

            This will overwrite the weights saved at this path (or the temporary
            weights).
        """
        if path is None:
            if not hasattr(self, "_weights_path"):
                # Use a random UUID to have a low probability of collisions
                # Use a hidden file so the user never notices how ugly that is
                self._weights_path = ".tmp-{}.pt".format(uuid4())
            path = self._weights_path

        checkpoint = self.state_dict()
        torch.save(checkpoint, path)
