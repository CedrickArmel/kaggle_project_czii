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
from collections.abc import Sequence
from typing import Any

import torch
from monai.utils import LossReduction

from .pooling import LocalFocalLoss


class DeepLoss(torch.nn.Module):
    def __init__(
        self,
        max_pool: "dict[str, Any]",
        avg_pool: "dict[str, Any]",
        num_classes: "int",
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
        depth: "int" = 3,
    ):

        losses = []

        for i in range(depth):
            if isinstance(weight, (int, float)):
                level_weight = (
                    [1] + [weight] * (num_classes - 1) if num_classes > 1 else None
                )
            elif isinstance(weight, Sequence):
                level_weight = (
                    [1] + [weight[i]] * (num_classes - 1) if num_classes > 1 else None
                )
            else:
                level_weight = None

            losses.append(
                LocalFocalLoss(
                    max_pool=max_pool,
                    avg_pool=avg_pool,
                    include_background=include_background,
                    to_onehot_y=to_onehot_y,
                    gamma=gamma,
                    alpha=alpha,
                    weight=level_weight,
                    reduction=reduction,
                    use_softmax=use_softmax,
                    lambda_max=lambda_max,
                    lambda_avg=lambda_avg,
                    weighted=weighted,
                )
            )

        self.level_losses = torch.nn.ModuleList(losses)
        self.depth = depth

    def forward(self, input: "torch.Tensor", target: "torch.Tensor", depth: "int"):
        return self.level_losses[depth](input=input, target=target)
