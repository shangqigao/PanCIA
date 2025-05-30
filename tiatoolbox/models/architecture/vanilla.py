"""Defines vanilla CNNs with torch backbones, mainly for patch classification."""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models

from tiatoolbox.models.models_abc import ModelABC
from tiatoolbox.utils.misc import select_device


def _get_architecture(arch_name, weights="DEFAULT", **kwargs):
    """Get a model.

    Model architectures are either already defined within torchvision or
    they can be custom-made within tiatoolbox.

    Args:
        arch_name (str):
            Architecture name.

    Returns:
        List of PyTorch network layers wrapped with `nn.Sequential`.
        https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html

    """
    backbone_dict = {
        "alexnet": torch_models.alexnet,
        "resnet18": torch_models.resnet18,
        "resnet34": torch_models.resnet34,
        "resnet50": torch_models.resnet50,
        "resnet101": torch_models.resnet101,
        "resnext50_32x4d": torch_models.resnext50_32x4d,
        "resnext101_32x8d": torch_models.resnext101_32x8d,
        "wide_resnet50_2": torch_models.wide_resnet50_2,
        "wide_resnet101_2": torch_models.wide_resnet101_2,
        "densenet121": torch_models.densenet121,
        "densenet161": torch_models.densenet161,
        "densenet169": torch_models.densenet169,
        "densenet201": torch_models.densenet201,
        "inception_v3": torch_models.inception_v3,
        "googlenet": torch_models.googlenet,
        "mobilenet_v2": torch_models.mobilenet_v2,
        "mobilenet_v3_large": torch_models.mobilenet_v3_large,
        "mobilenet_v3_small": torch_models.mobilenet_v3_small,
    }
    if arch_name not in backbone_dict:
        raise ValueError(f"Backbone `{arch_name}` is not supported.")

    creator = backbone_dict[arch_name]
    model = creator(weights=weights, **kwargs)

    # Unroll all the definition and strip off the final GAP and FCN
    if "resnet" in arch_name or "resnext" in arch_name:
        return nn.Sequential(*list(model.children())[:-2])
    if "densenet" in arch_name:
        return model.features
    if "alexnet" in arch_name:
        return model.features
    if "inception_v3" in arch_name or "googlenet" in arch_name:
        return nn.Sequential(*list(model.children())[:-3])

    return model.features


class CNNModel(ModelABC):
    """Retrieve the model backbone and attach an extra FCN to perform classification.

    Args:
        backbone (str):
            Model name.
        num_classes (int):
            Number of classes output by model.

    Attributes:
        num_classes (int):
            Number of classes output by the model.
        feat_extract (nn.Module):
            Backbone CNN model.
        pool (nn.Module):
            Type of pooling applied after feature extraction.
        classifier (nn.Module):
            Linear classifier module used to map the features to the
            output.

    """

    def __init__(self, backbone, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.feat_extract = _get_architecture(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Best way to retrieve channel dynamically is passing a small forward pass
        prev_num_ch = self.feat_extract(torch.rand([2, 3, 96, 96])).shape[1]
        self.classifier = nn.Linear(prev_num_ch, num_classes)

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(self, imgs):
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor):
                Model input.

        """
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        gap_feat = torch.flatten(gap_feat, 1)
        logit = self.classifier(gap_feat)
        return torch.softmax(logit, -1)

    @staticmethod
    def postproc(image):
        """Define the post-processing of this class of model.

        This simply applies argmax along last axis of the input.

        """
        return np.argmax(image, axis=-1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (ndarray):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        """
        img_patches_device = batch_data.to(select_device(on_gpu)).type(
            torch.float32
        )  # to NCHW
        img_patches_device = img_patches_device.permute(0, 3, 1, 2).contiguous()

        # Inference mode
        model.eval()
        # Do not compute the gradient (not training)
        with torch.inference_mode():
            output = model(img_patches_device)
        # Output should be a single tensor or scalar
        return output.cpu().numpy()


class CNNBackbone(ModelABC):
    """Retrieve the model backbone and strip the classification layer.

    This is a wrapper for pretrained models within pytorch.

    Args:
        backbone (str):
            Model name. Currently, the tool supports following
             model names and their default associated weights from pytorch.
                - "alexnet"
                - "resnet18"
                - "resnet34"
                - "resnet50"
                - "resnet101"
                - "resnext50_32x4d"
                - "resnext101_32x8d"
                - "wide_resnet50_2"
                - "wide_resnet101_2"
                - "densenet121"
                - "densenet161"
                - "densenet169"
                - "densenet201"
                - "inception_v3"
                - "googlenet"
                - "mobilenet_v2"
                - "mobilenet_v3_large"
                - "mobilenet_v3_small"

    Examples:
        >>> # Creating resnet50 architecture from default pytorch
        >>> # without the classification layer with its associated
        >>> # weights loaded
        >>> model = CNNBackbone(backbone="resnet50")
        >>> model.eval()  # set to evaluation mode
        >>> # dummy sample in NHWC form
        >>> samples = torch.rand(4, 3, 512, 512)
        >>> features = model(samples)
        >>> features.shape  # features after global average pooling
        torch.Size([4, 2048])

    """

    def __init__(self, backbone):
        super().__init__()
        self.feat_extract = _get_architecture(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    # pylint: disable=W0221
    # because abc is generic, this is actual definition
    def forward(self, imgs):
        """Pass input data through the model.

        Args:
            imgs (torch.Tensor):
                Model input.

        """
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        return torch.flatten(gap_feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        Contains logic for forward operation as well as i/o aggregation.

        Args:
            model (nn.Module):
                PyTorch defined model.
            batch_data (ndarray):
                A batch of data generated by
                `torch.utils.data.DataLoader`.
            on_gpu (bool):
                Whether to run inference on a GPU.

        """
        img_patches_device = batch_data.to(select_device(on_gpu)).type(
            torch.float32
        )  # to NCHW
        # img_patches_device = img_patches_device.permute(0, 3, 1, 2).contiguous()

        # Inference mode
        model.eval()
        # Do not compute the gradient (not training)
        with torch.inference_mode():
            output = model(img_patches_device)
        # Output should be a single tensor or scalar
        return [output.cpu().numpy()]
