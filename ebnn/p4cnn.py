"""Bayesian C4 group-equivariant CNN (P4CNN, Cohen & Welling 2016) -- equivariant by construction.

Unlike :class:`ebnn.models.BayesianGroupConvCNN` (the paper's ``gcnn`` method: untied
weights, equivariance imposed only at init by the orbit expansion), every layer here
stores only its *free* filters and builds the full weight by rotating / cyclically
shifting them.  The mean-field Gaussian posterior lives on the free filters, so each
posterior sample is an exactly C4-equivariant network and the KL is counted once per
free parameter.  Used by ``--method p4cnn`` as the "equivariant BNN" comparison.

Channel layout follows the rest of the package: a group feature map with ``C`` fields
has ``4 C`` channels, channel ``4 c + r`` holding field ``c`` at rotation ``r``.  With
cross-correlation, ``conv(rot x, rot^r psi) = rot conv(x, rot^(r-1) psi)``, so rotating
the input by 90 degrees rotates every map and shifts ``r -> r + 1`` within each field:

* lifting   ``out[c, r]    = conv(x, rot^r psi[c])``
* group     ``out[c, r]    = sum_{c', s} conv(f[c', s], rot^r psi[c, c', (s - r) mod 4])``
* read-out  mean over the 4 rotations of each field (invariant), then a Bayesian 1x1
  conv and global average pooling, as in :class:`ebnn.models.BayesianFullyConvCNN`.

2x2 max-pooling commutes with the rotations as long as the feature maps have even
side lengths (28 -> 14 -> 7 for FashionMNIST, 224 -> 7 for NCT-CRC-HE).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ebnn.layers import BayesianConv2dGaussian, gaussian_kl
from ebnn.symmetrize import rotate_kernel_c4

GROUP_SIZE = 4

# Same construction-time initialisation as the fully-/group-conv models in
# ``ebnn.models``; the trainer overwrites it (Kaiming mean, ``--rho_init`` scale).
_P4_GAUSSIAN_INIT = {"weight_mu_init": "zeros", "weight_rho_init": 0.693}


class _BayesianP4ConvGaussian(nn.Module):
    """Shared posterior / prior / sampling logic of the lifting and group convolutions.

    Exposes ``weight_mu`` / ``weight_rho`` / ``bias_mu`` / ``bias_rho`` / ``prior_mu`` /
    ``prior_std`` like :class:`ebnn.layers.BayesianConv2dGaussian`, so the posterior
    re-initialisation helpers and the optimiser builder in :mod:`ebnn.symmetrize`
    apply unchanged (``kaiming_normal_`` on the 5-D group filter uses fan-in
    ``C_in * 4 * k * k``, the fan-in of the expanded weight).
    """

    def __init__(
        self,
        free_weight_shape: Tuple[int, ...],
        out_fields: int,
        padding: int,
        prior_std: float = 1.0,
        rho_init: float = -3.0,
        random_prior: bool = False,
        weight_mu_init: str = "kaiming",
        weight_rho_init: Optional[float] = None,
    ):
        super().__init__()
        self.out_fields = out_fields
        self.out_channels = out_fields * GROUP_SIZE
        self.padding = padding

        if random_prior:
            self.register_buffer("prior_mu", torch.randn(free_weight_shape))
            self.register_buffer("prior_std", torch.full(free_weight_shape, float(prior_std)))
        else:
            self.prior_mu = 0.0
            self.prior_std = float(prior_std)

        self.weight_mu = nn.Parameter(torch.empty(free_weight_shape))
        if weight_mu_init == "kaiming":
            nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="relu")
        elif weight_mu_init == "zeros":
            nn.init.zeros_(self.weight_mu)
        else:
            raise ValueError(f"Unknown weight_mu_init: {weight_mu_init!r} (expected 'kaiming' or 'zeros')")
        w_rho = rho_init if weight_rho_init is None else weight_rho_init
        self.weight_rho = nn.Parameter(torch.full(free_weight_shape, float(w_rho)))

        # One bias per field, shared by its 4 rotations.
        self.bias_mu = nn.Parameter(torch.zeros(out_fields))
        self.bias_rho = nn.Parameter(torch.full((out_fields,), float(rho_init)))

    def expand_weight(self, free_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_std = F.softplus(self.weight_rho)
        free_weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

        bias_std = F.softplus(self.bias_rho)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        if isinstance(self.prior_std, torch.Tensor):
            bias_prior_std = self.prior_std.mean().detach()
        else:
            bias_prior_std = self.prior_std

        kl = gaussian_kl(self.weight_mu, weight_std, self.prior_mu, self.prior_std) + gaussian_kl(
            self.bias_mu, bias_std, 0.0, bias_prior_std
        )
        weight = self.expand_weight(free_weight)
        out = F.conv2d(x, weight, bias.repeat_interleave(GROUP_SIZE), padding=self.padding)
        return out, kl


class BayesianP4LiftingConvGaussian(_BayesianP4ConvGaussian):
    """Z2 -> P4 lifting convolution: ``in_channels`` planar maps -> ``out_fields`` x 4 rotations."""

    def __init__(self, in_channels: int, out_fields: int, kernel_size: int, padding: int = 0, **kwargs):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        super().__init__((out_fields, in_channels, kernel_size, kernel_size), out_fields, padding, **kwargs)

    def expand_weight(self, free_weight: torch.Tensor) -> torch.Tensor:
        # [C_out, C_in, k, k] -> [C_out, 4 (r), C_in, k, k] -> [4 C_out, C_in, k, k]
        rotated = torch.stack([rotate_kernel_c4(free_weight, r) for r in range(GROUP_SIZE)], dim=1)
        return rotated.reshape(self.out_channels, *free_weight.shape[1:])


class BayesianP4GroupConvGaussian(_BayesianP4ConvGaussian):
    """P4 -> P4 group convolution: ``in_fields`` x 4 -> ``out_fields`` x 4 channels."""

    def __init__(self, in_fields: int, out_fields: int, kernel_size: int, padding: int = 0, **kwargs):
        self.in_fields = in_fields
        self.in_channels = in_fields * GROUP_SIZE
        self.kernel_size = kernel_size
        super().__init__(
            (out_fields, in_fields, GROUP_SIZE, kernel_size, kernel_size), out_fields, padding, **kwargs
        )
        # _shift[r][s] = (s - r) mod 4: the free-filter rotation slot seen by input rotation s.
        self._shift = [[(s - r) % GROUP_SIZE for s in range(GROUP_SIZE)] for r in range(GROUP_SIZE)]

    def expand_weight(self, free_weight: torch.Tensor) -> torch.Tensor:
        # W[(c, r), (c', s)] = rot^r psi[c, c', (s - r) mod 4]
        # [C_out, C_in, 4 (s), k, k] -> [C_out, 4 (r), C_in, 4 (s), k, k] -> [4 C_out, 4 C_in, k, k]
        blocks = [rotate_kernel_c4(free_weight[:, :, self._shift[r]], r) for r in range(GROUP_SIZE)]
        k = free_weight.shape[-1]
        return torch.stack(blocks, dim=1).reshape(self.out_channels, self.in_channels, k, k)


class BayesianP4CNN(nn.Module):
    """[P4 conv -> ReLU -> Pool] x N -> rotation mean -> Conv1x1(num_classes) -> global average pool.

    ``base_channels`` are the numbers of fields per layer, so the feature maps have
    ``4 x base_channels`` channels -- the same widths as
    :class:`ebnn.models.BayesianGroupConvCNN` with the same ``base_channels``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        prior_params: Optional[dict] = None,
        base_channels: Optional[List[int]] = None,
        rho_init: float = -3.0,
        random_prior: bool = False,
    ):
        super().__init__()
        self.family = "gaussian"
        self.num_classes = num_classes
        self.group_size = GROUP_SIZE
        self.base_channels = base_channels or [64, 128, 256]
        if prior_params is None:
            prior_params = {"std": 1.0}
        conv_kwargs = {
            "prior_std": prior_params.get("std", 1.0),
            "rho_init": rho_init,
            "random_prior": random_prior,
            **_P4_GAUSSIAN_INIT,
        }

        self.conv_layers = nn.ModuleList()
        current_in = in_channels
        for idx, out_fields in enumerate(self.base_channels):
            layer_cls = BayesianP4LiftingConvGaussian if idx == 0 else BayesianP4GroupConvGaussian
            self.conv_layers.append(
                layer_cls(current_in, out_fields, kernel_size=3, padding=1, **conv_kwargs)
            )
            current_in = out_fields
        self.classifier = BayesianConv2dGaussian(
            current_in, num_classes, kernel_size=1, padding=0, **conv_kwargs
        )

    def forward(self, x, return_kl=True, return_probs=False):
        total_kl = torch.tensor(0.0, device=x.device)
        out = x
        for conv in self.conv_layers:
            out, kl = conv(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            total_kl = total_kl + kl
        b, c, h, w = out.shape
        out = out.view(b, c // GROUP_SIZE, GROUP_SIZE, h, w).mean(dim=2)  # rotation-invariant fields
        out, kl_last = self.classifier(out)
        total_kl = total_kl + kl_last
        out = F.adaptive_avg_pool2d(out, (1, 1))
        logits = out.squeeze(-1).squeeze(-1)
        probs = F.softmax(logits, dim=1) if return_probs else None
        if return_kl and return_probs:
            return logits, probs, total_kl
        if return_kl:
            return logits, total_kl
        if return_probs:
            return logits, probs
        return logits
