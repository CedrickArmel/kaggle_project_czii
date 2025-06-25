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

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class Score(Metric):
    def __init__(
        self,
        beta: "float" = 2.0,
        multiplier: "float" = 1.0,
        voxel_spacing: "float" = 10,
        **kwargs,
    ) -> "None":
        super().__init__(**kwargs)
        self.beta = beta
        self.multiplier = multiplier
        self.radius = {"1": 60, "2": 65, "3": 90, "4": 150, "5": 130, "6": 135}
        self.weights = {"1": 1, "2": 0, "3": 2, "4": 1, "5": 2, "6": 1}
        self.voxel_spacing = voxel_spacing
        self.add_state(name="preds", default=[], dist_reduce_fx="cat")
        self.add_state(name="targets", default=[], dist_reduce_fx="cat")
        self.preds: "list[torch.Tensor]"
        self.targets: "list[torch.Tensor]"

    def update(self, pred: "torch.Tensor", target: "torch.Tensor") -> "None":
        self.preds.append(pred)
        self.targets.append(target)

    def compute(self) -> "dict[str, float]":
        preds: "torch.Tensor" = dim_zero_cat(x=self.preds)  # z, y, x, label, run_id
        targets: "torch.Tensor" = dim_zero_cat(x=self.targets)
        targets = torch.unique(targets, dim=0).squeeze(dim=0)
        ths: "NDArray" = np.arange(start=0, stop=1.0, step=0.001)
        scores = []
        for t in ths:
            select = preds[:, -1] > t
            tpreds = preds[select][:, :-1]
            score = self.score(preds=tpreds, targets=targets)
            scores += [score]
        best_idx = int(np.argmax(a=scores))
        thd = float(ths[best_idx])
        fbeta = float(scores[best_idx])
        return dict(score=fbeta, thd=thd)

    def compute_metrics(
        self, candidates: "torch.Tensor", references: "torch.Tensor", radius: "float"
    ) -> "tuple[int, ...]":
        radius = radius * self.multiplier

        n_references = references.shape[0]
        n_candidates = candidates.shape[0]

        ref_tree = KDTree(references.cpu().numpy())
        candidate_tree = KDTree(candidates.cpu().numpy())
        raw_matches = candidate_tree.query_ball_tree(ref_tree, r=radius)
        raw_matches_extend = []
        for match in raw_matches:
            raw_matches_extend.extend(match)
        matches_within_threshold = set(raw_matches_extend)
        tp = int(len(matches_within_threshold))
        fp = int(n_candidates - tp)
        fn = int(n_references - tp)
        return tp, fp, fn

    def score(self, preds: "torch.Tensor", targets: "torch.Tensor"):
        select_candidates: "torch.Tensor" = torch.isin(preds[:, -1], targets[:, -1])
        preds = preds[select_candidates]
        runs = torch.unique(targets[:, -1]).cpu().numpy().tolist()
        particles = torch.unique(targets[:, -2]).cpu().numpy().tolist()
        results = {}
        for obj in particles:
            results[str(obj)] = {"tp": 0, "fp": 0, "fn": 0}

        for run in runs:
            for particle in particles:
                radius = int(self.radius[str(particle)] / self.voxel_spacing)
                select = (targets[:, -1] == run) & (targets[:, -2] == particle)
                references = targets[select]

                select = (preds[:, -1] == run) & (preds[:, -2] == particle)
                candidates = preds[select]

                if references.shape[0] == 0:
                    results[str(particle)]["fp"] += candidates.shape[0]
                    continue

                if candidates.shape[0] == 0:
                    results[str(particle)]["fn"] += references.shape[0]
                    continue

                tp, fp, fn = self.compute_metrics(
                    candidates=candidates, references=references, radius=radius
                )
                results[str(particle)]["tp"] += tp
                results[str(particle)]["fp"] += fp
                results[str(particle)]["fn"] += fn

        aggregate_fbeta = 0.0
        for particle, totals in results.items():
            tp = totals["tp"]
            fp = totals["fp"]
            fn = totals["fn"]

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            fbeta = (
                (1 + self.beta**2)
                * (precision * recall)
                / (self.beta**2 * precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            aggregate_fbeta += fbeta * self.weights.get(str(particle), 1.0)
        aggregate_fbeta / sum(self.weights.values())
        return aggregate_fbeta
