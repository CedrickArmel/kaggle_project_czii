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

import monai.transforms as mt


def get_transforms(
    mode: "str",
    ratios: "list[int]",
    roi_size: "list[int]",
    batch_size: "int",
    n_classes: "int",
) -> "mt.Compose":
    if mode not in ["static", "test", "train", "validation"]:
        raise ValueError("mode argument must be one of eval, static, test or train!")

    if mode.lower() == "static":
        compose = mt.Compose(
            [
                mt.EnsureChannelFirstd(
                    keys=["input", "target"], channel_dim="no_channel"
                ),
                mt.NormalizeIntensityd(keys=["input"]),
                mt.Orientationd(keys=["input", "target"], axcodes="RAS"),
            ]
        )
    elif mode.lower() == "train":
        compose = mt.Compose(
            [
                mt.RandCropByLabelClassesd(
                    keys=["input", "target"],
                    label_key="target",
                    spatial_size=list(roi_size),
                    num_samples=batch_size,
                    num_classes=n_classes,
                    ratios=ratios,
                    warn=True,
                )
            ]
        )
    elif mode.lower() == "validation":
        compose = mt.Compose(
            [
                mt.GridPatchd(
                    keys=["input", "target"],
                    patch_size=list(roi_size),
                    pad_mode="reflect",
                )
            ]
        )
    elif mode.lower() == "test":
        compose = mt.Compose(
            [
                mt.EnsureChannelFirstd(keys=["input"], channel_dim="no_channel"),
                mt.NormalizeIntensityd(keys=["input"]),
                mt.Orientationd(keys=["input"], axcodes="RAS"),
                mt.GridPatchd(
                    keys=["input"], patch_size=list(roi_size), pad_mode="reflect"
                ),
            ]
        )
    return compose
