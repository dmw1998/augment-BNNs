"""Bayesian (variational) layers for mean-field VI.

The Gaussian layers support two initialisations:

* ``weight_mu_init="kaiming", weight_rho_init=None`` -- Kaiming mean, ``rho_init`` scale
  (Theorem-2 / Theorem-4 models).
* ``weight_mu_init="zeros", weight_rho_init=0.693``  -- zero mean, ``softplus(0.693) ~= 1``
  scale (multi-family / fully-convolutional models).

All Gaussian layers expose the prior as ``prior_mu`` / ``prior_std`` (floats for an
isotropic prior, buffers for a random prior), as read by :mod:`ebnn.symmetrize`.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PriorStd = Union[float, torch.Tensor]


def gaussian_kl(
    mu: torch.Tensor,
    std: torch.Tensor,
    prior_mu: PriorStd,
    prior_std: PriorStd,
) -> torch.Tensor:
    """Analytic KL[ N(mu, std^2) || N(prior_mu, prior_std^2) ], summed over all elements."""
    var = std**2
    prior_var = prior_std**2
    kl = 0.5 * (torch.log(prior_var / var) + (var + (mu - prior_mu) ** 2) / prior_var - 1)
    return kl.sum()


def laplace_kl(mu: torch.Tensor, scale: torch.Tensor, prior_scale: float) -> torch.Tensor:
    """Analytic KL[ Laplace(mu, b) || Laplace(0, b0) ], summed over all elements."""
    eps = 1e-8
    b = scale.clamp_min(eps)
    b0 = torch.as_tensor(prior_scale, device=mu.device, dtype=mu.dtype).clamp_min(eps)
    delta = torch.abs(mu)
    kl = torch.log(b0 / b) + (delta + b * torch.exp(-delta / b)) / b0 - 1
    return kl.sum()


def _init_weight_mu(weight_mu: nn.Parameter, mode: str) -> None:
    if mode == "kaiming":
        nn.init.kaiming_normal_(weight_mu, mode="fan_in", nonlinearity="relu")
    elif mode == "zeros":
        nn.init.zeros_(weight_mu)
    else:
        raise ValueError(f"Unknown weight_mu_init: {mode!r} (expected 'kaiming' or 'zeros')")


# --------------------------------------------------------------------------- #
# Gaussian
# --------------------------------------------------------------------------- #
class BayesianLinearGaussian(nn.Module):
    """Bayesian linear layer with a mean-field Gaussian posterior."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        rho_init: float = -3.0,
        random_prior: bool = False,
        weight_mu_init: str = "kaiming",
        weight_rho_init: Optional[float] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight_shape = (out_features, in_features)
        if random_prior:
            self.register_buffer("prior_mu", torch.randn(weight_shape))
            self.register_buffer("prior_std", torch.full(weight_shape, float(prior_std)))
        else:
            self.prior_mu = 0.0
            self.prior_std = float(prior_std)

        self.weight_mu = nn.Parameter(torch.empty(weight_shape))
        _init_weight_mu(self.weight_mu, weight_mu_init)
        w_rho = rho_init if weight_rho_init is None else weight_rho_init
        self.weight_rho = nn.Parameter(torch.full(weight_shape, float(w_rho)))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), float(rho_init)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_std = F.softplus(self.weight_rho)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

        bias_std = F.softplus(self.bias_rho)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        if isinstance(self.prior_std, torch.Tensor):
            bias_prior_std = self.prior_std.mean().detach()
        else:
            bias_prior_std = self.prior_std

        kl = gaussian_kl(self.weight_mu, weight_std, self.prior_mu, self.prior_std) + gaussian_kl(
            self.bias_mu, bias_std, 0.0, bias_prior_std
        )
        return F.linear(x, weight, bias), kl


class BayesianConv2dGaussian(nn.Module):
    """Bayesian 2D convolution with a mean-field Gaussian posterior."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        prior_std: float = 1.0,
        rho_init: float = -3.0,
        random_prior: bool = False,
        weight_mu_init: str = "kaiming",
        weight_rho_init: Optional[float] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_shape = (out_channels, in_channels, kernel_size, kernel_size)

        if random_prior:
            self.register_buffer("prior_mu", torch.randn(self.weight_shape))
            self.register_buffer("prior_std", torch.full(self.weight_shape, float(prior_std)))
        else:
            self.prior_mu = 0.0
            self.prior_std = float(prior_std)

        self.weight_mu = nn.Parameter(torch.empty(self.weight_shape))
        _init_weight_mu(self.weight_mu, weight_mu_init)
        w_rho = rho_init if weight_rho_init is None else weight_rho_init
        self.weight_rho = nn.Parameter(torch.full(self.weight_shape, float(w_rho)))

        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_rho = nn.Parameter(torch.full((out_channels,), float(rho_init)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_std = F.softplus(self.weight_rho)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

        bias_std = F.softplus(self.bias_rho)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        if isinstance(self.prior_std, torch.Tensor):
            bias_prior_std = self.prior_std.mean().detach()
        else:
            bias_prior_std = self.prior_std

        kl = gaussian_kl(self.weight_mu, weight_std, self.prior_mu, self.prior_std) + gaussian_kl(
            self.bias_mu, bias_std, 0.0, bias_prior_std
        )
        out = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)
        return out, kl


# --------------------------------------------------------------------------- #
# Laplace
# --------------------------------------------------------------------------- #
class BayesianLinearLaplace(nn.Module):
    """Bayesian linear layer with a Laplace variational posterior."""

    def __init__(self, in_features: int, out_features: int, prior_scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_scale = prior_scale

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="relu")
        self.weight_log_scale = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_scale = nn.Parameter(torch.randn(out_features) * 0.1 - 5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_scale = F.softplus(self.weight_log_scale).clamp_min(1e-8)
        weight = torch.distributions.Laplace(self.weight_mu, weight_scale).rsample()

        bias_scale = F.softplus(self.bias_log_scale).clamp_min(1e-8)
        bias = torch.distributions.Laplace(self.bias_mu, bias_scale).rsample()

        kl = laplace_kl(self.weight_mu, weight_scale, self.prior_scale) + laplace_kl(
            self.bias_mu, bias_scale, self.prior_scale
        )
        return F.linear(x, weight, bias), kl


class BayesianConv2dLaplace(nn.Module):
    """Bayesian 2D convolution with a Laplace variational posterior."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        prior_scale: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_scale = prior_scale
        self.weight_shape = (out_channels, in_channels, kernel_size, kernel_size)

        self.weight_mu = nn.Parameter(torch.empty(self.weight_shape))
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="relu")
        self.weight_log_scale = nn.Parameter(torch.randn(self.weight_shape) * 0.1 - 5)

        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_log_scale = nn.Parameter(torch.randn(out_channels) * 0.1 - 5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_scale = F.softplus(self.weight_log_scale).clamp_min(1e-8)
        weight = torch.distributions.Laplace(self.weight_mu, weight_scale).rsample()

        bias_scale = F.softplus(self.bias_log_scale).clamp_min(1e-8)
        bias = torch.distributions.Laplace(self.bias_mu, bias_scale).rsample()

        kl = laplace_kl(self.weight_mu, weight_scale, self.prior_scale) + laplace_kl(
            self.bias_mu, bias_scale, self.prior_scale
        )
        out = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)
        return out, kl


# --------------------------------------------------------------------------- #
# Log-Normal
# --------------------------------------------------------------------------- #
# LogNormal only produces positive samples, so a fixed random sign buffer is
# applied to allow negative weights.
class BayesianLinearLogNormal(nn.Module):
    """Bayesian linear layer with a Log-Normal variational posterior."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        target_mu = 0.5 * np.log(2.0 / in_features)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1 + target_mu)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)

        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1 - 5)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1 - 5)

        self.register_buffer("weight_sign", torch.randint(0, 2, (out_features, in_features)).float() * 2 - 1)
        self.register_buffer("bias_sign", torch.randint(0, 2, (out_features,)).float() * 2 - 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_sigma = F.softplus(self.weight_rho)
        weight = torch.distributions.LogNormal(self.weight_mu, weight_sigma).rsample() * self.weight_sign

        bias_sigma = F.softplus(self.bias_rho)
        bias = torch.distributions.LogNormal(self.bias_mu, bias_sigma).rsample() * self.bias_sign

        kl = gaussian_kl(self.weight_mu, weight_sigma, self.prior_mu, self.prior_sigma) + gaussian_kl(
            self.bias_mu, bias_sigma, self.prior_mu, self.prior_sigma
        )
        return F.linear(x, weight, bias), kl


class BayesianConv2dLogNormal(nn.Module):
    """Bayesian 2D convolution with a Log-Normal variational posterior."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.weight_shape = (out_channels, in_channels, kernel_size, kernel_size)

        fan_in = in_channels * kernel_size * kernel_size
        target_mu = 0.5 * np.log(2.0 / fan_in)
        self.weight_mu = nn.Parameter(torch.randn(self.weight_shape) * 0.1 + target_mu)
        self.weight_rho = nn.Parameter(torch.randn(self.weight_shape) * 0.1 - 5)

        self.bias_mu = nn.Parameter(torch.randn(out_channels) * 0.1 - 5)
        self.bias_rho = nn.Parameter(torch.randn(out_channels) * 0.1 - 5)

        self.register_buffer("weight_sign", torch.randint(0, 2, self.weight_shape).float() * 2 - 1)
        self.register_buffer("bias_sign", torch.randint(0, 2, (out_channels,)).float() * 2 - 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_sigma = F.softplus(self.weight_rho)
        weight = torch.distributions.LogNormal(self.weight_mu, weight_sigma).rsample() * self.weight_sign

        bias_sigma = F.softplus(self.bias_rho)
        bias = torch.distributions.LogNormal(self.bias_mu, bias_sigma).rsample() * self.bias_sign

        kl = gaussian_kl(self.weight_mu, weight_sigma, self.prior_mu, self.prior_sigma) + gaussian_kl(
            self.bias_mu, bias_sigma, self.prior_mu, self.prior_sigma
        )
        out = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)
        return out, kl
