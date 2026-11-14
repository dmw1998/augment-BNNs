"""Orbit-averaged likelihood BNN (Nabarro et al., UAI 2022), C4-invariant by construction.

One posterior sample is applied to all four rotations of an input and the predictions
are averaged:

* ``prob``   p(y | x, theta) = (1/4) sum_g softmax_y f_theta(g x)
* ``logit``  p(y | x, theta) = softmax_y ((1/4) sum_g f_theta(g x))

:class:`OrbitAveragedBNN` wraps a network that samples one weight per forward pass
(e.g. :class:`ebnn.models.BayesianFullyConvCNN`), stacks the orbit into one batch of
``4 B`` images and returns logits whose softmax is the orbit-averaged predictive, so the
trainer and metrics apply unchanged.  Cost: 4x the forward passes of the base network.
"""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn

GROUP_SIZE = 4
ORBIT_LIKELIHOODS = ("prob", "logit")


def c4_orbit(x: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W) -> (4 B, C, H, W)``: rotations by 0 / 90 / 180 / 270 degrees, rotation-major."""
    return torch.cat([torch.rot90(x, r, dims=(2, 3)) for r in range(GROUP_SIZE)], dim=0)


class OrbitAveragedBNN(nn.Module):
    """Orbit-averaged predictive of a Bayesian base network (one weight sample per orbit).

    Same call API as :class:`ebnn.models.BayesianFullyConvCNN`
    (``forward(x, return_kl=True, return_probs=False)``); the KL is the base network's.
    """

    def __init__(self, base: nn.Module, likelihood: str = "prob"):
        super().__init__()
        if likelihood not in ORBIT_LIKELIHOODS:
            raise ValueError(f"Unknown orbit likelihood: {likelihood!r} (expected one of {ORBIT_LIKELIHOODS})")
        self.base = base
        self.likelihood = likelihood
        self.family = getattr(base, "family", "gaussian")
        self.num_classes = getattr(base, "num_classes", None)

    def forward(
        self, x: torch.Tensor, return_kl: bool = True, return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        b = x.shape[0]
        logits, kl = self.base(c4_orbit(x), return_kl=True)  # one theta for all 4 B images
        logits = logits.view(GROUP_SIZE, b, -1)
        if self.likelihood == "prob":
            # log((1/4) sum_g softmax f(g x)): softmax of this is the averaged probability
            out = torch.logsumexp(logits.log_softmax(dim=-1), dim=0) - math.log(GROUP_SIZE)
        else:
            out = logits.mean(dim=0)
        probs = out.softmax(dim=1) if return_probs else None
        if return_kl and return_probs:
            return out, probs, kl
        if return_kl:
            return out, kl
        if return_probs:
            return out, probs
        return out
