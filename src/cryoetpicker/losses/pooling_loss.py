# MIT License
#
# Copyright (c) 2025, Yebouet Cédrick-Armel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""from .focal_loss import sigmoid_focal_loss, softmax_focal_loss

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F
from monai.networks import one_hot
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss


class AvgPoolFocalLoss(_Loss):
    def __init__(self,
        max_pool_args: "dict[str, Any]",
        avg_pool_args: "dict[str, Any]",
        include_background: "bool" = True,
        to_onehot_y: "bool" = False, gamma: "float" = 2.0,
        downsample_y: "bool" = False
        alpha: "float | None" = None,  weight: "Sequence[float] | float | int | None" = None,
        contributions: "Sequence[float] | float | int | None" = None,
        reduction: "LossReduction | str" = LossReduction.MEAN,
        use_softmax: "bool" = False):

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        super().__init__(reduction=LossReduction(reduction).value)

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.downsample_y = downsample_y
        self.gamma = gamma
        self.alpha = alpha
        self.use_softmax = use_softmax
        weight = torch.tensor(weight) if weight is not None else None
        if weight is not None and weight.min() < 0:
            raise ValueError(
                "the value/values of the `weight` should be no less than 0."
            )
        contributions = torch.tensor(contributions) if contributions is not None else None
        if contributions is not None and contributions.min() < 0:
            raise ValueError(
                "the value/values of the `contributions` should be no less than 0."
            )
        self.register_buffer("class_weight", weight)
        self.register_buffer("contributions", contributions)
        self.class_weight: "torch.Tensor | None"
        self.contributions: "torch.Tensor | None"


    def forward(self, input: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":

        n_pred_ch: "int" = input.shape[1]

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise ValueError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )

        loss: Optional[torch.Tensor] = None
        input = input.float()
        target = target.float()

        if self.downsample_y:








class PoolingLossListModule("torch.nn.Module"):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        gamma: float = 2.0,
        alpha: float | None = None,
        weight: "Sequence[float] | float | int | None" = None,
        contributions: Sequence[float] | float | int | torch.Tensor | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        use_softmax: bool = False,
        max_pool_args: "dict[str, Any]",
        avg_pool_args: "dict[str, Any]"):

        self.focal = FocalLoss(**kwargs)
        self.register_buffer("contributions", torch.tensor())
        self.contribtions
"""
