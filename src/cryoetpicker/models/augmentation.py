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

import random

import torch
from torch import nn
from torch.distributions import Beta


class Mixup(nn.Module):  # type: ignore[misc]
    """Mixup augmentation for data."""

    def __init__(
        self, p: "float" = 0.0, beta: "float" = 5.0, add: "bool" = False
    ) -> None:
        """Initialize the Mixup module."""
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(beta, beta)
        self.mixadd = add
        self.p = p

    def forward(
        self, X: "torch.Tensor", Y: "torch.Tensor", Z: "torch.Tensor | None" = None
    ) -> "tuple[torch.Tensor, ...]":
        """Apply mixup augmentation to the input data."""
        bs = X.shape[0]
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        X_coeffs = coeffs.view((-1,) + (1,) * (X.ndim - 1))
        Y_coeffs = coeffs.view((-1,) + (1,) * (Y.ndim - 1))
        X = X_coeffs * X + (1 - X_coeffs) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y[perm]
        if Z:
            return X, Y, Z
        return X, Y


class CutmixSimple(nn.Module):
    """Simple cutmix augmentation for data."""

    def __init__(self, p: "float" = 0.0, beta: "float" = 5.0, dims: "tuple" = (-2, -1)):
        super().__init__()
        assert all(_ < 0 for _ in dims), "dims must be negatively indexed."
        self.beta_distribution = Beta(beta, beta)  # beta = 5 = gaussianlike
        self.dims = dims
        self.p = p

    def forward(self, X, Y, Z=None):
        cut_idx = self.beta_distribution.sample().item()

        perm = torch.randperm(X.size(0))
        X_perm = X[perm]
        Y_perm = Y[perm]

        axis = random.choice(self.dims)

        # Get cut idxs
        cutoff_X = int(cut_idx * X.shape[axis])
        cutoff_Y = int(cut_idx * Y.shape[axis])

        # Apply cut
        if axis == -1:
            X[..., :cutoff_X] = X_perm[..., :cutoff_X]
            Y[..., :cutoff_Y] = Y_perm[..., :cutoff_Y]
        elif axis == -2:
            X[..., :cutoff_X, :] = X_perm[..., :cutoff_X, :]
            Y[..., :cutoff_Y, :] = Y_perm[..., :cutoff_Y, :]
        else:
            raise ValueError("CutmixSimple: Axis not implemented.")

        return X, Y
