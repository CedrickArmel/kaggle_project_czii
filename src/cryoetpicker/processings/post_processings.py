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


def get_output_size(
    img: "torch.Tensor",
    locations: "torch.Tensor",
    roi_size: "torch.Tensor",
    device: "torch.device",
) -> "torch.Tensor":
    """Get the output size of the reconstructed image.
    Args:
        img (torch.Tensor): The input image tensor.
        locations (torch.Tensor): The locations of the detected points.
        roi_size (list): The size of the region of interest.
    Returns:
        torch.Tensor: The output size of the reconstructed image.
    """
    shapes = locations.max(2)[0]
    output_size = torch.zeros(5, device=device)
    s: "torch.Tensor" = torch.unique(shapes, dim=0).squeeze().to(device)
    s = s + roi_size
    output_size[0] = shapes.shape[0]
    output_size[1] = img.shape[1]
    output_size[2:] = s
    return output_size.int()


def reconstruct(
    img: "torch.Tensor",
    locations: "torch.Tensor",
    out_size: "torch.Tensor",
    roi_size: "torch.Tensor",
    device: "torch.device",
) -> "torch.Tensor":
    """Reconstruct the image from the detected points.
    Args:
        img (torch.Tensor): The input image tensor.
        locations (torch.Tensor): The locations of the detected points.
        out_size (torch.Tensor): The output size of the reconstructed image.
        roi_size (list): The size of the region of interest.
    Returns:
        torch.Tensor: The reconstructed image tensor.
    """
    reconstructed_img = torch.zeros(out_size.tolist(), device=device)
    reshape = list([locations.shape[0], locations.shape[2]]) + list(img.shape[1:])
    image = img.reshape(reshape)
    for i in range(out_size[0]):
        for j in range(locations.shape[2]):
            reconstructed_img[i][
                :,
                locations[i][0][j] : locations[i][0][j] + roi_size[0],
                locations[i][1][j] : locations[i][1][j] + roi_size[1],
                locations[i][2][j] : locations[i][2][j] + roi_size[2],
            ] = image[i][j, :]
    return reconstructed_img


def simple_nms(scores: "torch.Tensor", nms_radius: "int") -> "torch.Tensor":
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x: "torch.Tensor") -> "torch.Tensor":
        """Max pooling operation to find the maximum value in a local neighborhood."""
        return F.max_pool3d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)


def post_process_pipeline(
    input: "dict[str, Any]",
    nms_radius: "int",
    num_classes: "int",
    topk: "int",
    roi_size: "list[int]",
    tomo_size: "list[int]",
    interpolation: "str",
    align_corners: "bool | None" = None,
) -> "torch.Tensor":
    """Post-process the output of the model to get the final coordinates and confidence scores."""

    img: "torch.Tensor" = input["logits"].detach()
    if img.shape[-3:] != torch.Size(roi_size):
        img = F.interpolate(
            img,
            size=list(roi_size),
            mode="trilinear",
            align_corners=True,
        )

    device = img.device
    roi_size = torch.tensor(roi_size, device=device)
    locations: "torch.Tensor" = input["location"]
    tomo_ids: "torch.Tensor" = input["id"]

    out_size = get_output_size(img, locations, roi_size, device)
    rec_img = reconstruct(
        img=img,
        locations=locations,
        out_size=out_size,
        roi_size=roi_size,
        device=device,
    )

    s = torch.tensor(rec_img.shape[-3:], device=device)
    t = torch.tensor(tomo_size, device=device)

    delta = (s - t) // 2  # delta to remove padding added during transforms
    dz, dy, dx = delta.tolist()
    nz, ny, nx = t.tolist()

    rec_img = rec_img[:, :, dz : nz + dz, dy : ny + dy, dx : nx + dx]

    rec_img = F.interpolate(
        rec_img,
        size=[d // 2 for d in t.tolist()],
        mode=interpolation,
        align_corners=align_corners,
    )

    preds: "torch.Tensor" = rec_img.softmax(1)
    output: "torch.Tensor" = torch.empty(0, device=device)
    for i in range(1, num_classes):
        pred = preds[:, i, :][None,]
        nms: "torch.Tensor" = simple_nms(pred, nms_radius=nms_radius)  # (1,B, D, H, W)
        nms = nms.squeeze(dim=0)  # (B, D, H, W)

        flat_nms = nms.reshape(nms.shape[0], -1)  # (B, D*H*W)
        conf, indices = torch.topk(flat_nms, k=topk, dim=1)
        zyx = torch.stack(
            torch.unravel_index(indices, nms.shape[-3:]), dim=-1
        ).int()  # (B, K, 3)

        ids = torch.unique(tomo_ids.reshape(zyx.shape[0], -1), dim=1).expand(
            zyx.shape[0], topk
        )

        conf = conf.float()

        ids = ids.reshape(-1, 1)
        conf = conf.reshape(-1, 1)
        zyx = zyx.reshape(-1, 3)
        labels = torch.tensor([i] * zyx.shape[0], device=device).reshape(-1, 1)
        output = torch.cat([output, torch.cat([zyx, labels, ids, conf], dim=1)], dim=0)
    return output
