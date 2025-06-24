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

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F
from monai.networks import one_hot
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss

from .focal_loss import FocalLoss, sigmoid_focal_loss, softmax_focal_loss


class AvgPool3DLoss(_Loss):
    def __init__(
        self,
        kernel_size: "list[int] | int",
        stride: "list[int] | int",
        padding: "list[int] | int",
        include_background: "bool" = True,
        to_onehot_y: "bool" = False,
        gamma: "float" = 2.0,
        alpha: "float | None" = None,
        weight: "list[float] | float | int | None" = None,
        reduction: "LossReduction | str" = LossReduction.MEAN,
        use_softmax: "bool" = False,
    ):
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        super().__init__(reduction=LossReduction(reduction).value)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.gamma = gamma
        self.alpha = alpha
        self.use_softmax = use_softmax
        class_weight: "torch.Tensor | None" = (
            torch.tensor(weight) if weight is not None else None
        )
        if class_weight is not None and class_weight.min() < 0:
            raise ValueError(
                "the value/values of the `weight` should be no less than 0."
            )
        self.register_buffer("class_weight", class_weight)
        self.class_weight: "torch.Tensor | None"

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

        loss: "torch.Tensor | None" = None
        input = input.float()
        target = target.float()

        input = F.avg_pool3d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        target = F.avg_pool3d(
            target,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn(
                    "`include_background=False`, `alpha` ignored when using softmax."
                )
            loss = softmax_focal_loss(input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss(input, target, self.gamma, self.alpha)

        class_weight = self.class_weight
        num_of_classes = target.shape[1]

        if class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if class_weight.ndim == 0:
                class_weight = class_weight.repeat(num_of_classes)
            else:
                if class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            # apply class_weight to loss
            broadcast_dims: "list[int]" = [1, num_of_classes] + [1] * (loss.ndim - 2)
            class_weight = class_weight.view(broadcast_dims)
            loss = class_weight * loss

        loss = loss.mean(dim=list(range(2, target.ndim)))

        if self.reduction == LossReduction.SUM.value:
            loss = loss.sum()
        elif self.reduction == LossReduction.MEAN.value:
            loss = loss.mean()
        elif self.reduction == LossReduction.NONE.value:
            pass
        return loss


class MaxPool3DLoss(_Loss):
    def __init__(
        self,
        kernel_size: "list[int] | int",
        stride: "list[int] | int",
        padding: "list[int] | int",
        dilation: "list[int] | int",
        include_background: "bool" = True,
        to_onehot_y: "bool" = False,
        gamma: "float" = 2.0,
        alpha: "float | None" = None,
        weight: "list[float] | float | int | None" = None,
        reduction: "LossReduction | str" = LossReduction.MEAN,
        use_softmax: "bool" = False,
    ):
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        super().__init__(reduction=LossReduction(reduction).value)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.gamma = gamma
        self.alpha = alpha
        self.use_softmax = use_softmax
        class_weight: "torch.Tensor | None" = (
            torch.tensor(weight) if weight is not None else None
        )
        if class_weight is not None and class_weight.min() < 0:
            raise ValueError(
                "the value/values of the `weight` should be no less than 0."
            )
        self.register_buffer("class_weight", class_weight)
        self.class_weight: "torch.Tensor | None"

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

        loss: "torch.Tensor | None" = None
        input = input.float()
        target = target.float()

        input = F.max_pool3d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        target = F.max_pool3d(
            target,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn(
                    "`include_background=False`, `alpha` ignored when using softmax."
                )
            loss = softmax_focal_loss(input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss(input, target, self.gamma, self.alpha)

        class_weight = self.class_weight
        num_of_classes = target.shape[1]

        if class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if class_weight.ndim == 0:
                class_weight = class_weight.repeat(num_of_classes)
            else:
                if class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            # apply class_weight to loss
            broadcast_dims: "list[int]" = [1, num_of_classes] + [1] * (loss.ndim - 2)
            class_weight = class_weight.view(broadcast_dims)
            loss = class_weight * loss

        loss = loss.mean(dim=list(range(2, target.ndim)))

        if self.reduction == LossReduction.SUM.value:
            loss = loss.sum()
        elif self.reduction == LossReduction.MEAN.value:
            loss = loss.mean()
        elif self.reduction == LossReduction.NONE.value:
            pass
        return loss


class LocalFocalLoss(_Loss):
    def __init__(
        self,
        max_pool: "dict[str, Any]",
        avg_pool: "dict[str, Any]",
        include_background: "bool" = True,
        to_onehot_y: "bool" = False,
        gamma: "float" = 2.0,
        alpha: "float | None" = None,
        weight: "list[float] | float | int | None" = None,
        reduction: "LossReduction | str" = LossReduction.MEAN,
        use_softmax: "bool" = False,
        lambda_max: float = 0.25,
        lambda_avg: float = 0.25,
        weighted: "bool" = True,
    ):
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError("`reduction` must be mean or sum")

        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            alpha=alpha,
            weight=weight,
            reduction=reduction,
            use_softmax=use_softmax,
        )

        self.max_loss = MaxPool3DLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            alpha=alpha,
            weight=weight,
            reduction=reduction,
            use_softmax=use_softmax,
            **max_pool,
        )

        self.avg_loss = AvgPool3DLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            alpha=alpha,
            weight=weight,
            reduction=reduction,
            use_softmax=use_softmax,
            **avg_pool,
        )

        if lambda_max < 0.0:
            raise ValueError("lambda_max should be no less than 0.0.")
        if lambda_avg < 0.0:
            raise ValueError("lambda_avg should be no less than 0.0.")

        self.lambda_max = lambda_max
        self.lambda_avg = lambda_avg
        self.to_onehot_y = to_onehot_y

    def forward(self, input: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if input.shape[-3:] != target.shape[-3:]:
            target = F.adaptive_max_pool3d(target, input.shape[-3:])

        loss = self.focal(input, target)
        total_weight = 1.0
        if self.lambda_max > 0:
            loss += self.lambda_max * self.max_loss(input, target)
            total_weight += self.lambda_max
        if self.lambda_avg > 0:
            loss += self.lambda_avg * self.avg_loss(input, target)
            total_weight += self.lambda_avg
        if self.weighted:
            loss /= total_weight
        return loss
