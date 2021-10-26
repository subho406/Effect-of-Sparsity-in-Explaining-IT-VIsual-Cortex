"""
    Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
"""
from typing import Any

from nupic.torch.modules import SparseWeights2d
import torch
import torch.nn as nn


__all__ = ["AlexNet", "alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, sparsity=0,dropout: float = 0.5) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            SparseWeights2d(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11,stride=4,
                padding=2),sparsity=sparsity,allow_extremes=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SparseWeights2d(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5,
                padding=2),sparsity=sparsity,allow_extremes=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SparseWeights2d(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3,
                padding=1),sparsity=sparsity,allow_extremes=True),
            nn.ReLU(inplace=True),
            SparseWeights2d(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,
                padding=1),sparsity=sparsity,allow_extremes=True),
            nn.ReLU(inplace=True),
            SparseWeights2d(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                padding=1),sparsity=sparsity,allow_extremes=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    return model

