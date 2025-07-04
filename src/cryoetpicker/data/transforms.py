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

# TODO: Docstrings

import monai.transforms as mt


def get_transforms(
    mode: "str",
    keys: "list[str]",
    roi_size: "list[int]",
    num_samples: "int | None" = None,
    num_classes: "int | None" = None,
    ratios: "list[int] | None" = None,
) -> "mt.Compose":
    if mode not in ["test", "train", "validation"]:
        raise ValueError("mode argument must be one of eval, test or train!")
    static: "list[mt.Transform]" = [
        mt.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        mt.NormalizeIntensityd(keys=["input"]),
        mt.Orientationd(keys=keys, axcodes="RAS"),
    ]
    if mode == "train":
        train: "list[mt.Transform]" = [
            mt.RandCropByLabelClassesd(
                keys=keys,
                label_key="target",
                spatial_size=list(roi_size),
                num_samples=num_samples,
                num_classes=num_classes,
                ratios=ratios,
                warn=True,
            )
        ]
        compose: "list[mt.Transform]" = static + train
    else:
        val: "list[mt.Transform]" = [
            mt.GridPatchd(
                keys=keys,
                patch_size=list(roi_size),
                pad_mode="reflect",
            )
        ]
        compose = static + val
    compose += [mt.FromMetaTensord(keys=["input", "target"], allow_missing_keys=True)]
    return mt.Compose(compose)
