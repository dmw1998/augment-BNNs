"""C4 symmetrisation operators and related utilities.

Contains the three one-shot symmetrisation mechanisms applied to a trained
posterior, plus the optimiser construction and posterior re-initialisation
helpers used by the unified trainer:

* :func:`apply_one_shot_c4_projection` -- exact projection onto the C4-equivariant subspace.
* :func:`apply_one_shot_c4_posterior_average` -- C4 posterior averaging over the orbit (geometric product-of-experts or arithmetic moment matching).
* :func:`expand_small_model_to_gcnn` -- block-circulant expansion of a trained small model into a group-convolutional model.
"""

import math
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Rotation + softplus helpers
# --------------------------------------------------------------------------- #
def rotate_kernel_c4(weight: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate spatial kernel by ``k * 90`` degrees over the (H, W) axes."""
    return torch.rot90(weight, k=k, dims=[-2, -1])


def rotate_kernel(weight: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate spatial kernel by an arbitrary angle. Exact for right angles."""
    angle_mod = angle_deg % 360.0
    if abs(angle_mod - 0.0) < 1e-8:
        return weight
    if abs(angle_mod - 90.0) < 1e-8:
        return torch.rot90(weight, k=1, dims=[-2, -1])
    if abs(angle_mod - 180.0) < 1e-8:
        return torch.rot90(weight, k=2, dims=[-2, -1])
    if abs(angle_mod - 270.0) < 1e-8:
        return torch.rot90(weight, k=3, dims=[-2, -1])

    angle_rad = torch.tensor(angle_deg * math.pi / 180.0, dtype=weight.dtype, device=weight.device)
    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
        dtype=weight.dtype,
        device=weight.device,
    ).unsqueeze(0)
    n, c, h, w = weight.shape
    w_in = weight.view(n * c, 1, h, w)
    grid = F.affine_grid(theta.expand(n * c, -1, -1), w_in.shape, align_corners=True)
    rotated = F.grid_sample(w_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return rotated.view(n, c, h, w)


def inverse_softplus(y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Stable inverse of softplus: x = log(exp(y) - 1) for positive y."""
    y = y.clamp_min(eps)
    return torch.where(y > 20.0, y, y + torch.log1p(-torch.exp(-y)))


def _scale_rho_std(rho: torch.Tensor, factor: float, eps: float = 1e-12) -> torch.Tensor:
    """Scale std by ``factor`` in rho-parameterisation: sigma' = factor * sigma."""
    sigma = F.softplus(rho).clamp_min(eps)
    sigma_scaled = (sigma * factor).clamp_min(eps)
    return inverse_softplus(sigma_scaled, eps=eps)


# --------------------------------------------------------------------------- #
# Method: proj -- exact projection onto the C4-equivariant subspace
# --------------------------------------------------------------------------- #
def _project_group_block(weight: torch.Tensor, group_in_start: int, group_out_start: int) -> None:
    device, dtype = weight.device, weight.dtype
    k_h, k_w = weight.shape[-2], weight.shape[-1]
    newweight = torch.zeros((4, k_h, k_w), device=device, dtype=dtype)
    for jj in range(4):
        for ii in range(4):
            k = group_out_start + ((-ii) % 4)
            j = group_in_start + ((jj - ii) % 4)
            newweight[jj, :, :] += 0.25 * rotate_kernel_c4(weight[k, j, :, :], ii)
    for kk in range(4):
        for jj in range(4):
            k = group_out_start + kk
            j = group_in_start + jj
            weight[k, j, :, :] = rotate_kernel_c4(newweight[(jj - kk) % 4, :, :], kk)


def _project_first_layer_block(weight: torch.Tensor, group_out_start: int, input_idx: int = 0) -> None:
    device, dtype = weight.device, weight.dtype
    k_h, k_w = weight.shape[-2], weight.shape[-1]
    newweight = torch.zeros((k_h, k_w), device=device, dtype=dtype)
    for ii in range(4):
        j = group_out_start + ((-ii) % 4)
        newweight += 0.25 * rotate_kernel_c4(weight[j, input_idx, :, :], ii)
    for kk in range(4):
        k = group_out_start + kk
        weight[k, input_idx, :, :] = rotate_kernel_c4(newweight, kk)


def _project_conv_weight_tensor(weight: torch.Tensor, first_layer: bool) -> torch.Tensor:
    out_ch, in_ch, _, _ = weight.shape
    projected = weight.clone()
    if first_layer:
        if out_ch % 4 != 0 or in_ch < 1:
            return projected
        for g in range(out_ch // 4):
            _project_first_layer_block(projected, group_out_start=4 * g, input_idx=0)
        return projected
    if out_ch % 4 != 0 or in_ch % 4 != 0:
        return projected
    for g_in in range(in_ch // 4):
        for g_out in range(out_ch // 4):
            _project_group_block(projected, group_in_start=4 * g_in, group_out_start=4 * g_out)
    return projected


def _project_sigma_tensor(rho: torch.Tensor, first_layer: bool) -> torch.Tensor:
    sigma = F.softplus(rho)
    sigma_projected = _project_conv_weight_tensor(sigma, first_layer=first_layer)
    eps = 1e-12
    sigma_projected = sigma_projected.clamp_min(eps)
    return inverse_softplus(sigma_projected, eps=eps)


def _project_bias_mu_block(bias_mu: torch.Tensor) -> None:
    for start in range(0, bias_mu.shape[0], 4):
        bias_mu[start : start + 4] = bias_mu[start : start + 4].mean()


def apply_one_shot_c4_projection(model: nn.Module) -> None:
    """Project posterior mean and scale onto the C4-equivariant subspace (in place)."""
    with torch.no_grad():
        conv_layers: List[nn.Module] = list(getattr(model, "conv_layers", []))
        for idx, layer in enumerate(conv_layers):
            if not (hasattr(layer, "weight_mu") and isinstance(layer.weight_mu, nn.Parameter)):
                continue
            if layer.weight_mu.dim() != 4 or layer.weight_mu.shape[-1] < 3:
                continue

            projected_mu = _project_conv_weight_tensor(layer.weight_mu.data, first_layer=(idx == 0))
            layer.weight_mu.data.copy_(projected_mu)

            if hasattr(layer, "weight_rho") and isinstance(layer.weight_rho, nn.Parameter):
                projected_rho = _project_sigma_tensor(layer.weight_rho.data, first_layer=(idx == 0))
                layer.weight_rho.data.copy_(projected_rho)

            if hasattr(layer, "bias_mu") and isinstance(layer.bias_mu, nn.Parameter):
                _project_bias_mu_block(layer.bias_mu.data)
            if hasattr(layer, "bias_rho") and isinstance(layer.bias_rho, nn.Parameter):
                _project_bias_mu_block(layer.bias_rho.data)

        clf = getattr(model, "classifier", None)
        if clf is not None and hasattr(clf, "weight_mu"):
            w = clf.weight_mu.data  # (C_out, 4*base, kH, kW)
            c_out, c_in = w.shape[:2]
            assert c_in % 4 == 0
            w_view = w.view(c_out, c_in // 4, 4, *w.shape[2:])
            w_view.copy_(w_view.mean(dim=2, keepdim=True).expand_as(w_view))


# --------------------------------------------------------------------------- #
# Method: avg -- one-shot C4 posterior averaging over the orbit
# --------------------------------------------------------------------------- #
def _gaussian_arithmetic_moment_match(mu_list, rho_list, eps: float = 1e-8):
    mus = torch.stack(mu_list, dim=0)
    sigmas = torch.stack([F.softplus(rho).clamp_min(eps) for rho in rho_list], dim=0)
    vars_ = sigmas.pow(2)
    mu_bar = mus.mean(dim=0)
    second_moment = (vars_ + mus.pow(2)).mean(dim=0)
    var_bar = (second_moment - mu_bar.pow(2)).clamp_min(eps)
    sigma_bar = torch.sqrt(var_bar)
    return mu_bar, inverse_softplus(sigma_bar, eps=eps)


def _gaussian_geometric_poe(mu_list, rho_list, eps: float = 1e-8):
    mus = torch.stack(mu_list, dim=0)
    sigmas = torch.stack([F.softplus(rho).clamp_min(eps) for rho in rho_list], dim=0)
    vars_ = sigmas.pow(2)
    precisions = 1.0 / vars_
    precision_tilde = precisions.mean(dim=0)
    var_tilde = 1.0 / precision_tilde.clamp_min(eps)
    mu_tilde = var_tilde * (precisions * mus).mean(dim=0)
    sigma_tilde = torch.sqrt(var_tilde.clamp_min(eps))
    return mu_tilde, inverse_softplus(sigma_tilde, eps=eps)


def apply_one_shot_c4_posterior_average(
    model: nn.Module,
    method: Literal["geometric", "arithmetic"] = "geometric",
) -> None:
    """One-shot C4 posterior averaging (in place). Skips the 1x1 classifier (kernel < 3)."""
    if method not in ("geometric", "arithmetic"):
        raise ValueError(f"Unknown averaging method: {method}")
    with torch.no_grad():
        for module in model.modules():
            if not hasattr(module, "weight_mu") or not isinstance(module.weight_mu, nn.Parameter):
                continue
            if module.weight_mu.dim() != 4 or module.weight_mu.shape[-1] < 3:
                continue
            if hasattr(module, "weight_rho") and isinstance(module.weight_rho, nn.Parameter):
                mu_rot = [rotate_kernel_c4(module.weight_mu.data, k) for k in range(4)]
                rho_rot = [rotate_kernel_c4(module.weight_rho.data, k) for k in range(4)]
                if method == "geometric":
                    mu_avg, rho_avg = _gaussian_geometric_poe(mu_rot, rho_rot)
                else:
                    mu_avg, rho_avg = _gaussian_arithmetic_moment_match(mu_rot, rho_rot)
                module.weight_mu.data.copy_(mu_avg)
                module.weight_rho.data.copy_(rho_avg)
            else:
                mu_avg = sum(rotate_kernel_c4(module.weight_mu.data, k) for k in range(4)) / 4.0
                module.weight_mu.data.copy_(mu_avg)


# --------------------------------------------------------------------------- #
# Method: gcnn -- block-circulant expansion of a trained small model
# --------------------------------------------------------------------------- #
def _outer_rotation_index(arrangement: str, g_out: int, group_size: int) -> int:
    """Outer rotation exponent alpha(i) for the orbit-expansion filter arrangements.

    With inner index (g_out - g_in), the net rotation of block (i, j) is
    R^{alpha(i) + i - j}, matching the appendix ablation:

    * "gcnn"         alpha(i) = +i  -> net R^{2i - j}  (default; used in the paper's main results)
    * "row_rolled"   alpha(i) =  0  -> net R^{i - j}
    * "row_constant" alpha(i) = -i  -> net R^{-j}      (all output rows identical)
    """
    if arrangement == "gcnn":
        return g_out % group_size
    if arrangement == "row_rolled":
        return 0
    if arrangement == "row_constant":
        return (-g_out) % group_size
    raise ValueError(
        f"Unknown arrangement: {arrangement!r} (expected 'gcnn', 'row_rolled', or 'row_constant')"
    )


def expand_small_model_to_gcnn(
    small_model: nn.Module,
    gcnn_model: nn.Module,
    group_size: int = 4,
    arrangement: str = "gcnn",
) -> None:
    """Expand small-model weights to a GCNN using C_|G| rotations.

    ``arrangement`` selects the filter-tiling scheme compared in the orbit-expansion
    ablation (see :func:`_outer_rotation_index`); the default ``"gcnn"`` is the
    block-circulant scheme used in the main results.
    """
    angles = [360.0 * g / group_size for g in range(group_size)]
    inv_sqrt_g = 1.0 / math.sqrt(group_size)

    with torch.no_grad():
        for layer_idx, (small_conv, gcnn_conv) in enumerate(
            zip(small_model.conv_layers, gcnn_model.conv_layers)
        ):
            if not hasattr(small_conv, "weight_mu"):
                continue

            base_weight_mu = small_conv.weight_mu.data
            base_weight_rho = small_conv.weight_rho.data if hasattr(small_conv, "weight_rho") else None
            is_first_layer = layer_idx == 0

            if is_first_layer:
                expanded_mu = []
                expanded_rho = [] if base_weight_rho is not None else None
                for g in range(group_size):
                    expanded_mu.append(rotate_kernel(base_weight_mu, angle_deg=angles[g]))
                    if base_weight_rho is not None:
                        expanded_rho.append(rotate_kernel(base_weight_rho, angle_deg=angles[g]))
                gcnn_conv.weight_mu.data = torch.cat(expanded_mu, dim=0)
                if expanded_rho is not None:
                    gcnn_conv.weight_rho.data = torch.cat(expanded_rho, dim=0)
            else:
                expanded_mu = []
                expanded_rho = [] if base_weight_rho is not None else None
                for g_out in range(group_size):
                    row_blocks_mu = []
                    row_blocks_rho = [] if base_weight_rho is not None else None
                    outer = _outer_rotation_index(arrangement, g_out, group_size)
                    for g_in in range(group_size):
                        k = (g_out - g_in) % group_size
                        rotated_mu = rotate_kernel(base_weight_mu, angle_deg=angles[k])
                        rotated_mu = rotate_kernel(rotated_mu, angle_deg=angles[outer])
                        row_blocks_mu.append(rotated_mu)
                        if base_weight_rho is not None:
                            rotated_rho = rotate_kernel(base_weight_rho, angle_deg=angles[k])
                            rotated_rho = rotate_kernel(rotated_rho, angle_deg=angles[outer])
                            row_blocks_rho.append(rotated_rho)
                    expanded_mu.append(torch.cat(row_blocks_mu, dim=1))
                    if base_weight_rho is not None:
                        expanded_rho.append(torch.cat(row_blocks_rho, dim=1))
                mu_expanded = torch.cat(expanded_mu, dim=0)
                gcnn_conv.weight_mu.data = mu_expanded * inv_sqrt_g
                if expanded_rho is not None:
                    rho_expanded = torch.cat(expanded_rho, dim=0)
                    gcnn_conv.weight_rho.data = _scale_rho_std(rho_expanded, inv_sqrt_g)

            if hasattr(small_conv, "bias_mu"):
                gcnn_conv.bias_mu.data = small_conv.bias_mu.data.repeat(group_size)
                if hasattr(small_conv, "bias_rho"):
                    gcnn_conv.bias_rho.data = small_conv.bias_rho.data.repeat(group_size)

        if hasattr(small_model.classifier, "weight_mu"):
            small_classifier_mu = small_model.classifier.weight_mu.data
            small_classifier_rho = (
                small_model.classifier.weight_rho.data
                if hasattr(small_model.classifier, "weight_rho")
                else None
            )
            expanded_classifier_mu = small_classifier_mu.repeat(1, group_size, 1, 1) / group_size
            gcnn_model.classifier.weight_mu.data = expanded_classifier_mu
            if small_classifier_rho is not None:
                expanded_classifier_rho = small_classifier_rho.repeat(1, group_size, 1, 1)
                gcnn_model.classifier.weight_rho.data = _scale_rho_std(
                    expanded_classifier_rho, 1.0 / group_size
                )
            if hasattr(small_model.classifier, "bias_mu"):
                gcnn_model.classifier.bias_mu.data = small_model.classifier.bias_mu.data.clone()
                if hasattr(small_model.classifier, "bias_rho"):
                    gcnn_model.classifier.bias_rho.data = small_model.classifier.bias_rho.data.clone()


# --------------------------------------------------------------------------- #
# Posterior (re-)initialisation
# --------------------------------------------------------------------------- #
def initialize_posterior_mean(model: nn.Module, posterior_init: str = "zero"):
    if posterior_init not in ("zero", "kaiming", "prior"):
        raise ValueError(f"Unknown posterior_init mode: {posterior_init}")
    for module in model.modules():
        if hasattr(module, "weight_mu") and isinstance(module.weight_mu, nn.Parameter):
            if posterior_init == "zero":
                nn.init.zeros_(module.weight_mu)
            elif posterior_init == "kaiming":
                if module.weight_mu.dim() >= 2:
                    nn.init.kaiming_normal_(module.weight_mu, mode="fan_in", nonlinearity="relu")
                else:
                    nn.init.zeros_(module.weight_mu)
            else:  # "prior"
                prior_mu = getattr(module, "prior_mu", 0.0)
                prior_std = getattr(module, "prior_std", 1.0)
                with torch.no_grad():
                    if isinstance(prior_mu, torch.Tensor):
                        if prior_mu.shape == module.weight_mu.shape:
                            module.weight_mu.copy_(
                                prior_mu.to(module.weight_mu.device, module.weight_mu.dtype)
                            )
                        else:
                            module.weight_mu.fill_(float(prior_mu.mean().item()))
                    else:
                        module.weight_mu.fill_(float(prior_mu))
                    if hasattr(module, "weight_rho") and isinstance(module.weight_rho, nn.Parameter):
                        eps = 1e-12
                        if isinstance(prior_std, torch.Tensor):
                            if prior_std.shape == module.weight_rho.shape:
                                target_std = prior_std.to(module.weight_rho.device, module.weight_rho.dtype)
                            else:
                                target_std = torch.full_like(
                                    module.weight_rho, float(prior_std.mean().item())
                                )
                        else:
                            target_std = torch.full_like(module.weight_rho, float(prior_std))
                        target_std = target_std.clamp_min(eps)
                        module.weight_rho.copy_(inverse_softplus(target_std, eps=eps))
        if hasattr(module, "bias_mu") and isinstance(module.bias_mu, nn.Parameter):
            nn.init.zeros_(module.bias_mu)
        if (
            posterior_init == "prior"
            and hasattr(module, "bias_rho")
            and isinstance(module.bias_rho, nn.Parameter)
        ):
            prior_std = getattr(module, "prior_std", 1.0)
            eps = 1e-12
            with torch.no_grad():
                bias_prior_std = (
                    float(prior_std.mean().item())
                    if isinstance(prior_std, torch.Tensor)
                    else float(prior_std)
                )
                bias_prior_std = max(bias_prior_std, eps)
                module.bias_rho.fill_(float(inverse_softplus(torch.tensor(bias_prior_std), eps=eps).item()))


def initialize_gaussian_posterior_scale(model: nn.Module, rho_init: float):
    for module in model.modules():
        if hasattr(module, "weight_rho") and isinstance(module.weight_rho, nn.Parameter):
            with torch.no_grad():
                module.weight_rho.fill_(rho_init)
        if hasattr(module, "bias_rho") and isinstance(module.bias_rho, nn.Parameter):
            with torch.no_grad():
                module.bias_rho.fill_(rho_init)


# --------------------------------------------------------------------------- #
# Optimiser construction + buffer reset
# --------------------------------------------------------------------------- #
def _is_rho(param_name: str) -> bool:
    return param_name.endswith("rho")  # weight_rho / bias_rho


def build_optimizer(
    model,
    optimizer: str = "adamw",
    lr: float = 1e-3,
    weight_decay: Optional[float] = None,
    momentum: float = 0.9,
    nesterov: bool = False,
    rho_lr_mult: float = 10.0,
    rho_lr: Optional[float] = None,
    rho_weight_decay: float = 0.0,
):
    """AdamW: single group. SGD: separate, larger-LR rho group (weight_decay=0)."""
    optimizer = optimizer.lower()
    if optimizer == "adamw":
        wd = 1e-3 if weight_decay is None else weight_decay
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if optimizer == "sgd":
        wd = 1e-4 if weight_decay is None else weight_decay
        rho_params, base_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (rho_params if _is_rho(name) else base_params).append(p)
        eff_rho_lr = rho_lr if rho_lr is not None else lr * rho_lr_mult
        groups = [
            {"params": base_params, "lr": lr, "weight_decay": wd},
            {"params": rho_params, "lr": eff_rho_lr, "weight_decay": rho_weight_decay},
        ]
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {optimizer!r} (expected 'adamw' or 'sgd')")


def clear_optimizer_state(optimizer) -> None:
    """Wipe per-parameter optimiser buffers (momentum / Adam m,v / step count)."""
    optimizer.state.clear()


def describe_optimizer(optimizer) -> str:
    parts = []
    for i, g in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in g["params"])
        parts.append(f"group{i}: lr={g.get('lr')}, wd={g.get('weight_decay')}, params={n}")
    return type(optimizer).__name__ + " [" + "; ".join(parts) + "]"
