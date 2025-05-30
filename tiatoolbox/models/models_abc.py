"""Defines Abstract Base Class for Models defined in tiatoolbox."""
from abc import ABC, abstractmethod

import torch.nn as nn


class IOConfigABC(ABC):
    """Define an abstract class for holding predictor I/O information.

    Enforcing such that following attributes must always be defined by
    the subclass.

    """

    @property
    @abstractmethod
    def input_resolutions(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def output_resolutions(self):
        raise NotImplementedError


class ModelABC(ABC, nn.Module):
    """Abstract base class for models used in tiatoolbox."""

    def __init__(self):
        super().__init__()
        self._postproc = self.postproc
        self._preproc = self.preproc

    @abstractmethod
    # noqa
    # This is generic abc, else pylint will complain
    def forward(self, *args, **kwargs):
        """Torch method, this contains logic for using layers defined in init."""
        ...  # pragma: no cover

    @staticmethod
    @abstractmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        Contains logic for forward operation as well as I/O aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (ndarray):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        """
        ...  # pragma: no cover

    @staticmethod
    def preproc(image):
        """Define the pre-processing of this class of model."""
        return image

    @staticmethod
    def postproc(image):
        """Define the post-processing of this class of model."""
        return image

    @property
    def preproc_func(self):
        """Return the current pre-processing function of this instance."""
        return self._preproc

    @preproc_func.setter
    def preproc_func(self, func):
        """Set the pre-processing function for this instance.

        If `func=None`, the method will default to `self.preproc`.
        Otherwise, `func` is expected to be callable.

        Examples:
            >>> # expected usage
            >>> # model is a subclass object of this ModelABC
            >>> # `func` is a user defined function
            >>> model = ModelABC()
            >>> model.preproc_func = func
            >>> transformed_img = model.preproc_func(img)

        """
        if func is not None and not callable(func):
            raise ValueError(f"{func} is not callable!")

        if func is None:
            self._preproc = self.preproc
        else:
            self._preproc = func

    @property
    def postproc_func(self):
        """Return the current post-processing function of this instance."""
        return self._postproc

    @postproc_func.setter
    def postproc_func(self, func):
        """Set the pre-processing function for this instance of model.

        If `func=None`, the method will default to `self.postproc`.
        Otherwise, `func` is expected to be callable and behave as
        follows:

        Examples:
            >>> # expected usage
            >>> # model is a subclass object of this ModelABC
            >>> # `func` is a user defined function
            >>> model = ModelABC()
            >>> model.postproc_func = func
            >>> transformed_img = model.postproc_func(img)

        """
        if func is not None and not callable(func):
            raise ValueError(f"{func} is not callable!")

        if func is None:
            self._postproc = self.postproc
        else:
            self._postproc = func
