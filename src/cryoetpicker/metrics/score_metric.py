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

"""
Derived from:
https://www.kaggle.com/code/metric/czi-cryoet-84969?scriptVersionId=208227222&cellId=1
"""

import torch
from torchmetrics import Metric

# from torchmetrics.utilities import dim_zero_cat


class Score(Metric):
    def __init__(
        self,
        beta: "float" = 2.0,
        multiplier: "float" = 1.0,
        radius: "float" = 500.0,
        threshold: "float" = 0.5,
        **kwargs,
    ) -> "None":
        super().__init__(**kwargs)
        self.beta = beta
        self.multiplier = multiplier
        self.radius = radius
        self.threshold = threshold
        self.add_state(name="preds", default=[], dist_reduce_fx="cat")
        self.add_state(name="targets", default=[], dist_reduce_fx="cat")
        self.preds: "list[torch.Tensor]"
        self.targets: "list[torch.Tensor]"

    def update(self, pred: "torch.Tensor", target: "torch.Tensor") -> "None":
        self.preds.append(pred)
        self.targets.append(target)

    def compute(self) -> "dict[str, float]":
        # preds: "torch.Tensor" = dim_zero_cat(x=self.preds)
        # targets: "torch.Tensor" = dim_zero_cat(x=self.targets)
        score = 0.0
        return dict(score=score)
