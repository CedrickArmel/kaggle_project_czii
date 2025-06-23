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

# MIT License
#
# Copyright (c) 2024, Yebouet Cédrick-Armel
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

# mypy: disable-error-code="misc, assignment"

import datetime
import os
from zoneinfo import ZoneInfo

import hydra
from lightning.pytorch.loggers import TensorBoardLogger
from monai.data import CacheDataset
from omegaconf import DictConfig, OmegaConf

from cryoetpicker.data import get_transforms, load_data
from cryoetpicker.models import LightningFxUnet3D
from cryoetpicker.trainers import get_lightning_trainer
from cryoetpicker.utils import get_callbacks, get_data_loader, get_profiler, set_seed

OmegaConf.register_new_resolver("eval", resolver=eval, replace=True)


@hydra.main(config_path="./config", config_name="config")
def train(cfg: "DictConfig") -> "None":
    set_seed(**cfg.determinism)

    train_data, val_data = load_data(**cfg.copick.base, **cfg.copick.fit)
    train_ds = CacheDataset(
        data=train_data,
        transform=get_transforms(**cfg.transforms.base, **cfg.transforms.train),
    )
    val_ds = CacheDataset(
        data=val_data,
        transform=get_transforms(**cfg.transforms.base, **cfg.transforms.validation),
    )
    train_loader = get_data_loader(
        dataset=train_ds, seed=cfg.determinism.seed, **cfg.loader.train
    )
    val_loader = get_data_loader(
        dataset=val_ds, seed=cfg.determinism.seed, **cfg.loader.eval
    )

    start_time = datetime.datetime.now(ZoneInfo("Europe/Paris")).strftime(
        "%Y%m%d%H%M%S"
    )
    save_dir = os.path.join(cfg.save_dir, f"seed_{cfg.determinism.seed}")
    cfg.trainers.lightning.default_root_dir = os.path.join(
        save_dir, f"fold{cfg.copick.fit.fold}", f"{start_time}"
    )

    os.makedirs(cfg.trainers.lightning.default_root_dir, exist_ok=True)
    chckpt_cb, lr_cb = get_callbacks(cfg.callbacks)

    model = LightningFxUnet3D(cfg)
    profiler = get_profiler(**cfg.profiler)

    logger = (
        TensorBoardLogger(save_dir=save_dir, name=f"fold{cfg.fold}")
        if cfg.trainers.callable.logger
        else cfg.trainers.callable.logger
    )
    if logger is not None:
        print("Logger log_dir : ", logger.log_dir)
    else:
        print("No logger registred.")

    callbacks = (
        [chckpt_cb, lr_cb]
        if cfg.trainers.callable.callbacks
        else cfg.trainers.callable.callbacks
    )
    trainer = get_lightning_trainer(
        logger=logger, callbacks=callbacks, profiler=profiler, **cfg.trainers
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.training.ckpt_path,
    )


if __name__ == "__main__":
    train()
