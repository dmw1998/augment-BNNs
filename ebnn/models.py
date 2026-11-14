"""Model definitions used across the experiments.

* :class:`BayesianCNN` -- conv-stack + FC head, Gaussian posterior.  Used by the Theorem-2 and Theorem-4 baselines (Kaiming mean init).
* :class:`BayesianCNNConfigurable` -- conv-stack + FC head with a selectable variational family (Gaussian / Laplace / Log-Normal).  Used by the multi-family comparison.
* :class:`BayesianFullyConvCNN` -- fully convolutional (1x1 conv classifier + global average pool), Gaussian posterior.
* :class:`BayesianGroupConvCNN` -- group-conv variant with ``|G|x`` channels; weights are not tied during training (equivariance is imposed only at init by func `ebnn.symmetrize.expand_small_model_to_gcnn`).
"""

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ebnn.layers import (
    BayesianConv2dGaussian,
    BayesianConv2dLaplace,
    BayesianConv2dLogNormal,
    BayesianLinearGaussian,
    BayesianLinearLaplace,
)
from ebnn.symmetrize import rotate_kernel_c4

# Construction-time init (zero mean, softplus(0.693) ~= 1 scale); the trainer
# re-initialises the posterior afterwards.
_FULLYCONV_GAUSSIAN_INIT = {"weight_mu_init": "zeros", "weight_rho_init": 0.693}

# Multi-family model (not re-initialised by its trainer): Kaiming mean + small std;
# a zero mean with std ~= 1 would start exactly at the N(0, 1) prior.
_CONFIGURABLE_GAUSSIAN_INIT = {"weight_mu_init": "kaiming", "weight_rho_init": None}


class BayesianCNN(nn.Module):
    """Conv-stack + FC head with a Gaussian posterior (Kaiming mean init)."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_channels: List[int],
        input_size: Tuple[int, int],
        prior_std: float = 1.0,
        rho_init: float = -3.0,
        random_prior: bool = False,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        c_in = in_channels
        for c_out in conv_channels:
            self.conv_layers.append(
                BayesianConv2dGaussian(
                    c_in,
                    c_out,
                    kernel_size=3,
                    padding=1,
                    prior_std=prior_std,
                    rho_init=rho_init,
                    random_prior=random_prior,
                )
            )
            c_in = c_out

        h, w = input_size
        with torch.no_grad():
            test = torch.randn(1, in_channels, h, w)
            for conv in self.conv_layers:
                test, _ = conv(test)
                test = F.relu(test)
                test = F.max_pool2d(test, 2)
            self.fc_input_size = test.view(1, -1).shape[1]

        self.fc = BayesianLinearGaussian(
            self.fc_input_size,
            num_classes,
            prior_std=prior_std,
            rho_init=rho_init,
            random_prior=random_prior,
        )

    def forward(self, x: torch.Tensor, return_kl: bool = True):
        kl_total = 0.0
        out = x
        for conv in self.conv_layers:
            out, kl = conv(out)
            kl_total = kl_total + kl
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out, kl = self.fc(out)
        kl_total = kl_total + kl
        if return_kl:
            return out, kl_total
        return out


class BayesianCNNConfigurable(nn.Module):
    """Conv-stack + FC head with a selectable variational family.

    Architecture: ``[Conv -> ReLU -> Pool] x N -> FC``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        family: Literal["gaussian", "laplace", "lognormal"] = "gaussian",
        prior_params: Optional[dict] = None,
        conv_channels: Optional[List[int]] = None,
        input_size: Tuple[int, int] = (28, 28),
        rho_init: float = -3.0,
        random_prior: bool = False,
    ):
        super().__init__()
        self.family = family
        self.input_size = input_size
        self.rho_init = rho_init
        self.random_prior = random_prior
        self.num_classes = num_classes

        if prior_params is None:
            prior_params = (
                {"std": 1.0}
                if family == "gaussian"
                else {"scale": 1.0} if family == "laplace" else {"mu": 0.0, "sigma": 1.0}
            )
        self.prior_params = prior_params

        if conv_channels is None:
            conv_channels = [64, 128, 256]
        self.conv_channels = conv_channels

        if family == "gaussian":
            conv_cls = BayesianConv2dGaussian
            conv_kwargs = {
                "prior_std": prior_params.get("std", 1.0),
                "rho_init": rho_init,
                "random_prior": random_prior,
                **_CONFIGURABLE_GAUSSIAN_INIT,
            }
        elif family == "laplace":
            conv_cls = BayesianConv2dLaplace
            conv_kwargs = {"prior_scale": prior_params.get("scale", 1.0)}
            if random_prior:
                print("Warning: random_prior is not implemented for the Laplace family.")
        elif family == "lognormal":
            conv_cls = BayesianConv2dLogNormal
            conv_kwargs = {
                "prior_mu": prior_params.get("mu", 0.0),
                "prior_sigma": prior_params.get("sigma", 1.0),
            }
            if random_prior:
                print("Warning: random_prior is not implemented for the LogNormal family.")
        else:
            raise ValueError(f"Unknown family: {family}")

        self.conv_layers = nn.ModuleList()
        current_in = in_channels
        for out_channels in conv_channels:
            self.conv_layers.append(
                conv_cls(current_in, out_channels, kernel_size=3, padding=1, **conv_kwargs)
            )
            current_in = out_channels

        self.fc_input_size = None
        self.fc = None
        self._initialize_fc(in_channels, input_size)

    def _initialize_fc(self, in_channels: int, input_size: Tuple[int, int]):
        h, w = input_size
        test_input = torch.randn(1, in_channels, h, w)
        with torch.no_grad():
            out = test_input
            for conv in self.conv_layers:
                out, _ = conv(out)
                out = F.relu(out)
                out = F.max_pool2d(out, 2)
            self.fc_input_size = out.view(1, -1).shape[1]
        print(f"Computed FC input size: {self.fc_input_size} (spatial: {out.shape[2]}x{out.shape[3]})")

        # The final layer is always Gaussian so that logits can be negative.
        if self.family == "laplace":
            self.fc = BayesianLinearLaplace(
                self.fc_input_size,
                self.num_classes,
                prior_scale=self.prior_params.get("scale", 1.0),
            )
        else:
            prior_std = self.prior_params.get("std", 1.0) if self.family == "gaussian" else 1.0
            self.fc = BayesianLinearGaussian(
                self.fc_input_size,
                self.num_classes,
                prior_std=prior_std,
                rho_init=self.rho_init,
                random_prior=self.random_prior,
                **_CONFIGURABLE_GAUSSIAN_INIT,
            )

    def forward(self, x: torch.Tensor, return_kl: bool = True, return_separate_kl: bool = False):
        conv_kl = torch.tensor(0.0, device=x.device)
        out = x
        for conv in self.conv_layers:
            out, kl = conv(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            conv_kl = conv_kl + kl
        out = out.view(out.size(0), -1)
        out, fc_kl = self.fc(out)

        if not return_kl:
            return out
        if return_separate_kl:
            return out, conv_kl, fc_kl
        return out, conv_kl + fc_kl


class BayesianFullyConvCNN(nn.Module):
    """[Conv -> ReLU -> Pool] x N -> Conv1x1(num_classes) -> global average pool."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        prior_params: Optional[dict] = None,
        conv_channels: Optional[List[int]] = None,
        rho_init: float = -3.0,
        random_prior: bool = False,
    ):
        super().__init__()
        self.family = "gaussian"
        self.num_classes = num_classes
        if prior_params is None:
            prior_params = {"std": 1.0}
        if conv_channels is None:
            conv_channels = [64, 128, 256]
        conv_kwargs = {
            "prior_std": prior_params.get("std", 1.0),
            "rho_init": rho_init,
            "random_prior": random_prior,
            **_FULLYCONV_GAUSSIAN_INIT,
        }

        self.conv_layers = nn.ModuleList()
        current_in = in_channels
        for out_channels in conv_channels:
            self.conv_layers.append(
                BayesianConv2dGaussian(current_in, out_channels, kernel_size=3, padding=1, **conv_kwargs)
            )
            current_in = out_channels
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

    def get_first_layer_rotations(self, filter_idx: int = 0):
        layer = self.conv_layers[0]
        w = layer.weight_mu.detach().cpu()[filter_idx]
        return {
            "k0": w,
            "k1": rotate_kernel_c4(w, 1),
            "k2": rotate_kernel_c4(w, 2),
            "k3": rotate_kernel_c4(w, 3),
        }


class BayesianGroupConvCNN(nn.Module):
    """Group-conv variant with ``|G|x`` channels.

    Weights are NOT tied during training; equivariance is imposed only at init
    by :func:`ebnn.symmetrize.expand_small_model_to_gcnn`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        prior_params: Optional[dict] = None,
        base_channels: Optional[List[int]] = None,
        group_size: int = 4,
        rho_init: float = -3.0,
        random_prior: bool = False,
    ):
        super().__init__()
        self.family = "gaussian"
        self.num_classes = num_classes
        self.group_size = group_size
        self.base_channels = base_channels or [64, 128, 256]
        if prior_params is None:
            prior_params = {"std": 1.0}
        conv_kwargs = {
            "prior_std": prior_params.get("std", 1.0),
            "rho_init": rho_init,
            "random_prior": random_prior,
            **_FULLYCONV_GAUSSIAN_INIT,
        }

        self.conv_layers = nn.ModuleList()
        current_in = in_channels
        for base_out_channels in self.base_channels:
            out_channels = base_out_channels * group_size
            self.conv_layers.append(
                BayesianConv2dGaussian(current_in, out_channels, kernel_size=3, padding=1, **conv_kwargs)
            )
            current_in = out_channels
        final_in = self.base_channels[-1] * group_size
        self.classifier = BayesianConv2dGaussian(
            final_in, num_classes, kernel_size=1, padding=0, **conv_kwargs
        )

    def forward(self, x, return_kl=True, return_probs=False):
        total_kl = torch.tensor(0.0, device=x.device)
        out = x
        for conv in self.conv_layers:
            out, kl = conv(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            total_kl = total_kl + kl
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
