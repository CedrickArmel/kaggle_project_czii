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
import shutil
from typing import Any
from warnings import warn

import copick
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm


def move_overlay(user_id: "str", session_id: "str", force: "bool" = False) -> "None":
    """Move overlay data then add user_id to filename

    Args:
        user_id (str): Copick user ID.
        session_id (str): Copick Session ID.
        force (bool, optional): Whether to force move if files already exist. Defaults to False.
    """
    source_dir = "/kaggle/input/czii-cryo-et-object-identification/train/overlay"
    destination_dir = "/kaggle/working/train/overlay"
    print(f"Copying files to {destination_dir} ...")
    if os.path.exists(destination_dir) and not force:
        warn(
            f"The directory '{destination_dir}' already exists. Set force to True to force overwriting. Exiting..."
        )
    else:
        for root, dirs, files in tqdm(os.walk(source_dir)):
            relative_path = os.path.relpath(root, source_dir)
            target_dir = os.path.join(destination_dir, relative_path)
            os.makedirs(target_dir, exist_ok=True)
            for file in files:
                new_filename = f"{user_id}_{session_id}_{file}"
                source_file = os.path.join(root, file)
                destination_file = os.path.join(target_dir, new_filename)
                shutil.copy2(source_file, destination_file)
        print("Copying task completed succesfully!")


def load_data(
    mode: "str",
    path: "str",
    fold: "str | None" = None,
    user_id: "str" = "copick",
    session_id: "str" = "0",
    repeat: "list[int]" = [2, 2],
    voxel_spacing: "int" = 10,
    resolution: "str" = "0",
    tomotype: "str" = "denoised",
) -> "tuple[list[dict[str, Any]], ...] | list[dict[str, Any]]":
    """Load training/inference data from filesystem"""
    if mode not in ["fit", "test"]:
        raise ValueError(
            f"mode argument ({mode}) is not valid. valid values are `fit`and `test`."
        )

    root = copick.from_file(path)

    if mode == "fit":
        mapping = dict(
            TS_5_4=0, TS_6_4=1, TS_6_6=2, TS_69_2=3, TS_73_6=4, TS_86_3=5, TS_99_9=6
        )
        eval_data = []
        train_data = []
        move_overlay(user_id=user_id, session_id=session_id)
    else:
        test_data = []

    for i, run in tqdm(enumerate(root.runs)):
        tomogram = (
            run.get_voxel_spacing(voxel_spacing)
            .get_tomograms(tomotype)[0]
            .numpy(resolution)
        )
        if mode == "fit":
            run_id = mapping[run.name]
            D, H, W = tomogram.shape
            segmentation = np.zeros((D, H, W), dtype="uint8")
            zyx_: "list[list[int]]" = []
            for obj in root.pickable_objects:
                pick = run.get_picks(object_name=obj.name, user_id=user_id)
                if len(pick):
                    for point in pick[0].points:
                        z = int(np.ceil(point.location.z / voxel_spacing))
                        y = int(np.ceil(point.location.y / voxel_spacing))
                        x = int(np.ceil(point.location.x / voxel_spacing))
                        segmentation[z, y, x] = int(obj.label)
                        zyx_.append([z, y, x, int(obj.label), run_id])
            zyx: "NDArray" = np.array(zyx_)
            if run.name == fold:
                eval_data.append(
                    {
                        "input": tomogram,
                        "target": segmentation,
                        "id": run_id,
                        "zyx": zyx,
                    }
                )
            else:
                train_data.append({"input": tomogram, "target": segmentation})
        else:
            test_data.append({"input": tomogram, "id": i})

    return (
        (train_data * repeat[0], eval_data * repeat[1])
        if mode == "fit"
        else test_data * repeat[0]
    )
