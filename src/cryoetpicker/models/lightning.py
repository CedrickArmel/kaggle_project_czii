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

import os
from typing import Any

import lightning.pytorch as L
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.utilities import dim_zero_cat

from cryoetpicker.losses import DeepLoss, DiceLoss
from cryoetpicker.metrics import Score
from cryoetpicker.processings import post_process_pipeline
from cryoetpicker.utils import get_optimizer, get_scheduler, initialize_weights

from .augmentation import CutmixSimple, Mixup
from .flexible_unet import FxFlexibleUNet


class LightningFxUnet3D(L.LightningModule):
    """Feature Extended Flexible 3D Unet Lightning Module"""

    def __init__(self, cfg: "DictConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.validation_step_outputs: "list[torch.Tensor]" = []
        self.backbone = FxFlexibleUNet(**cfg.models.flexible)
        self.mixup = Mixup(**cfg.training.augs.mixup)
        self.cutmix = CutmixSimple(**cfg.training.augs.cutmix)
        self.deep_loss = DeepLoss(**cfg.deep_loss)
        self.dice = DiceLoss(**cfg.dice)
        self.weighted = cfg.training.supervision.weighted
        self.train_sub_batch = cfg.training.sub_batch.train
        self.eval_sub_batch = cfg.training.sub_batch.eval
        deep_contibutions = torch.tensor(
            list(cfg.training.supervision.deep_contributions)
        )
        self.register_buffer("deep_contibutions", deep_contibutions)
        self.deep_contibutions: "torch.Tensor"

    def forward(self, batch: "dict[str, Any]") -> "dict[str, Any]":
        """Perform a forward pass through the model."""
        bs = self.train_sub_batch if self.training else self.eval_sub_batch
        has_target = "target" in batch
        full_size = batch["input"].shape[0]

        if has_target:
            target: "torch.Tensor" = batch["target"].float()
            if target.ndim > 5:
                target = target.squeeze(dim=0)

        all_outs = []
        outputs = {}

        for i in range(0, full_size, bs if bs != -1 else full_size):
            x: "torch.Tensor " = (
                batch["input"][i : i + (bs if bs != -1 else full_size)]
                .float()
            )
            if x.ndim > 5:
                x = x.squeeze(dim=0)
            if self.training:  # we assume a target is always present during training
                outs: "list[torch.Tensor]" = self.backbone(x)
                logits: "torch.Tensor" = outs[-1]
                all_outs.append(outs)
            else:
                with torch.no_grad():
                    outs = self.backbone(x)
                    logits = outs[-1]
                    all_outs.append(logits)

        if self.training:
            outs = [
                torch.cat([out[i] for out in all_outs]) for i in range(len(all_outs[0]))
            ]
            logits = outs[-1]

            loss = self.deep_contibutions[-1] * self.deep_loss(
                input=logits, target=target, depth=-1
            )

            for i in range(2, self.deep_loss.depth + 1):
                if self.deep_contibutions[-i] > 0:
                    x = outs[-i]
                    loss += self.deep_contibutions[-i] * self.deep_loss(
                        input=x, target=target, depth=-i
                    )

            if self.weighted:
                loss /= self.deep_contibutions.sum()

        else:
            with torch.no_grad():
                # For inference, we concatenate all outputs
                logits = torch.cat(all_outs, dim=0)
                if has_target:
                    loss = self.deep_loss(input=logits, target=target, depth=-1)

        if has_target:
            outputs["loss"] = loss
            outputs["dice"] = self.dice(logits, target).detach()

        if not self.training:
            outputs["logits"] = logits
            outputs["location"] = batch["input_meta_dict"]["location"]
            outputs["id"] = batch["id"]
        return outputs

    def setup(self, stage: "str") -> "None":
        """Called at the beginning of each stage in oder to build model dynamically."""
        freeze_encoder = self.cfg.training.supervision.freeze_encoder
        stepping_batches = self.trainer.estimated_stepping_batches
        max_epochs = self.cfg.trainers.lightning.max_epochs
        world_size = self.trainer.world_size

        self.cfg.lr *= self.trainer.world_size
        self.training_steps = stepping_batches * max_epochs * world_size

        if stage == "fit":
            if freeze_encoder:
                self.backbone.decoder = self.backbone.decoder.apply(
                    lambda m: initialize_weights(module=m, **self.cfg.w_init)
                )

                for param in self.backbone.encoder.parameters():
                    param.requires_grad = False
            else:
                self.backbone = self.backbone.apply(
                    lambda m: initialize_weights(module=m, **self.cfg.w_init)
                )

    def configure_optimizers(self) -> "dict[str, Any] | Optimizer":
        """Return the optimizer and an optionnal lr_scheduler"""
        optimizer: "Optimizer" = get_optimizer(self.backbone, **self.cfg.optimizer)
        scheduler: "LRScheduler | None" = get_scheduler(
            optimizer=optimizer,
            training_steps=self.training_steps,
            **self.cfg.scheduler,
        )
        return (
            dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler, interval="step", frequency=1, name="lr"
                ),
            )
            if scheduler is not None
            else optimizer
        )

    def training_step(
        self, batch: "dict[str, Any]", batch_idx: "int"
    ) -> "torch.Tensor":
        output_dict = self(batch)
        loss: "torch.Tensor" = output_dict["loss"]
        dice: "torch.Tensor" = output_dict["dice"].mean(dim=0).squeeze()
        log_dict: "dict[str, torch.Tensor]" = dict(
            train_loss=loss,
            train_bg_dice=dice[0],
            train_fg_dice=dice[1:].mean(),
        )
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: "dict[str, Any]", batch_idx: "int"
    ) -> "torch.Tensor":
        """Operates on a single batch of data from the validation set"""
        zyx = batch["zyx"]
        output_dict = self(batch)
        loss: "torch.Tensor" = output_dict["loss"]
        dice: "torch.Tensor" = output_dict["dice"].mean(dim=0).squeeze()
        preds: "torch.Tensor" = post_process_pipeline(input=output_dict, **self.cfg.pp)
        log_dict: "dict[str, torch.Tensor]" = dict(
            val_loss=loss,
            val_bg_dice=dice[0],
            val_fg_dice=dice[1:].mean(),
        )
        self.validation_step_outputs.append(preds)
        self.score_metric.update(preds, zyx)
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        return preds

    def on_fit_start(self) -> "None":
        """Called at the very beginning of fit."""
        if torch.distributed.is_initialized():
            if not hasattr(self, "gloo_group"):
                self.gloo_groupg = torch.distributed.new_group(backend="gloo")
                self.score_metric = Score(
                    process_group=self.gloo_groupg,
                    **self.cfg.metrics.score.init,
                    **self.cfg.metrics.score.super,
                )
        else:
            self.score_metric = Score(
                **self.cfg.metrics.score.init, **self.cfg.metrics.score.super
            )

    def on_train_batch_end(self, outputs, batch, batch_idx) -> "None":
        params = [p for p in self.parameters() if p.grad is not None]
        if len(params) == 0:
            total_norm = torch.tensor(0.0)
        else:
            total_norm = torch.norm(
                torch.cat([p.detach().view(-1) for p in params]),
                p=2,
            )
        self.log(
            "weight_norm",
            total_norm,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> "None":
        """Called after the epoch ends to agg preds and logging"""
        metrics = self.score_metric.compute()
        preds = dim_zero_cat(self.validation_step_outputs)
        preds = preds.cpu()
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        torch.save(
            preds,
            os.path.join(
                self.cfg.trainers.lightning.default_root_dir,
                f"val_epoch_{self.current_epoch}_end_step{self.global_step}_rank{self.global_rank}.pt",
            ),
        )
        self.score_metric.reset()
        self.validation_step_outputs.clear()

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: "float | None" = None,
        gradient_clip_algorithm: "Any | None" = None,
    ) -> "None":
        """Gradient clipping and tracking before/afater clipping"""
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            total_norm_before = torch.tensor(0.0)
        else:
            total_norm_before = torch.norm(
                torch.cat([g.detach().view(-1) for g in grads]),
                p=2,
            )

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            total_norm_after = torch.tensor(0.0)
        else:
            total_norm_after = torch.norm(
                torch.cat([g.detach().view(-1) for g in grads]),
                p=2,
            )

        log_dict: "dict[str, torch.Tensor]" = dict(
            grad_norm=total_norm_before, clip_grad_norm=total_norm_after
        )

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
        )
