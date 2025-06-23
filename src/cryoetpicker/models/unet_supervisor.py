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

from typing import Any

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from omegaconf import DictConfig
from torch import nn

from cryoetpicker.losses import FocalLoss

from .augmentation import CutmixSimple, Mixup


class UNetSupervisor(nn.Module):  # type: ignore[misc]
    """Adapted from ChristofHenkel/kaggle-cryoet-1st-place-segmentation/models
    to support sub_batches and avoid OOM errors.
    """

    def __init__(self, backbone: "torch.nn.Module", cfg: "DictConfig") -> "None":
        """Initialize the Net module."""
        # TODO: agnostic way to do multilevel supervision with a module list
        super(UNetSupervisor, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.mixup = Mixup(**cfg.training.augs.mixup)
        self.cutmix = CutmixSimple(**cfg.training.augs.cutmix)
        self.loss_fn = FocalLoss(**cfg.loss)
        self.dice_fn = DiceLoss(**cfg.dice)
        self.max_loss_fn = FocalLoss(**cfg.max_loss.loss)
        self.avg_loss_fn = FocalLoss(**cfg.avg_loss.loss)
        self.deep_supervision = cfg.training.supervision.deep_supervision
        self.weighted_loss = cfg.training.supervision.weighted_loss
        self.batches = cfg.training.batches
        loss_contributions = torch.tensor(
            list(cfg.training.supervision.loss_contributions)
        )
        self.register_buffer("loss_contributions", loss_contributions)
        self.loss_contributions: "torch.Tensor"

    def forward(self, batch: "dict[str, Any]") -> "dict[str, Any]":
        """Perform a forward pass through the model."""
        bs = self.batches.train if self.training else self.batches.eval
        has_target = "target" in batch
        full_size = batch["input"].shape[0]
        device: "torch.device" = batch["input"].device
        location = batch["input"].meta["location"]

        target: "torch.Tensor" = torch.empty(
            0, device=device
        )  # better than empty list because of XLA compilation performance
        all_outs = []
        outputs = {}

        for i in range(0, full_size, bs if bs != -1 else full_size):
            x: "torch.Tensor " = batch["input"][i : i + bs].float()
            y: "torch.Tensor | None" = (
                batch["target"][i : i + bs].float() if has_target else None
            )

            if self.training:  # we assume a target is always present during training
                outs: "list[torch.Tensor]" = self.backbone(x)
                logits: "torch.Tensor" = outs[-1]
                all_outs.append(outs)
                y = F.adaptive_max_pool3d(y, logits.shape[-3:])
                target = torch.cat([target, y], dim=0)
            else:
                with torch.no_grad():
                    outs = self.backbone(x)
                    logits = outs[-1]
                    all_outs.append(logits)
                    if has_target:
                        y = F.adaptive_max_pool3d(y, logits.shape[-3:])
                        target = torch.cat([target, y], dim=0)

        if self.training:
            outs = [
                torch.cat([out[i] for out in all_outs]) for i in range(len(all_outs[0]))
            ]
            logits = outs[-1]
            loss = self.loss_contributions[-1] * self.loss_fn(logits, target)
            x_aux: "torch.Tensor" = F.max_pool3d(logits, **self.cfg.max_loss.pooling)
            y_aux: "torch.Tensor" = F.max_pool3d(target, **self.cfg.max_loss.pooling)
            loss += self.loss_contributions[-2] * self.max_loss_fn(x_aux, y_aux)

            if self.deep_supervision:
                x_aux = outs[-2]
                y_aux = F.avg_pool3d(target, **self.cfg.avg_loss.pooling)
                loss += self.loss_contributions[-3] * self.avg_loss_fn(x_aux, y_aux)

            if self.weighted_loss:
                loss /= self.loss_contributions.sum()

        else:
            with torch.no_grad():
                # For inference, we concatenate all outputs
                logits = torch.cat(all_outs, dim=0)
                if has_target:
                    loss = self.loss_fn(logits, target)

        if has_target:
            outputs["loss"] = loss
            outputs["dice"] = self.dice_fn(logits, target).squeeze().mean(dim=0)

        if not self.training:
            outputs["logits"] = logits
            outputs["location"] = location
            if "id" in batch:
                outputs["id"] = batch["id"]
        return outputs
