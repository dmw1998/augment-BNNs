"""Unified trainer for C4-symmetrised Bayesian CNNs (mean-field Gaussian VI).

``--method``:

* ``avg``      -- one-shot C4 posterior averaging (geometric PoE / arithmetic).
* ``proj``     -- one-shot projection onto the C4-equivariant subspace.
* ``gcnn``     -- orbit expansion of a trained small model into a group-conv model.
* ``baseline`` -- augmented training only.
* ``enforce``  -- ``--enforce_op`` proj / avg at init, re-applied after every step.
* ``p4cnn``    -- Bayesian C4 G-CNN, equivariant by construction (:mod:`ebnn.p4cnn`).
* ``orbit``    -- orbit-averaged likelihood of Nabarro et al. (2022) (:mod:`ebnn.orbit`).

``avg`` / ``proj`` are triggered by ``--trigger_acc`` or ``--trigger_epoch``; optimiser
buffers are then cleared, and ``--sym_rescale`` restores every weight tensor to its
pre-symmetrisation norm.  ``--conv_channels`` is the width of every method (``gcnn``:
stage-2 width, stage 1 uses ``conv_channels // group_size``; ``p4cnn``:
``conv_channels // 4`` fields x 4 rotations).

Example:
    python scripts/train_symmetrization.py                 # gcnn + SGD, FashionMNIST, N0 = 5000
    python scripts/train_symmetrization.py --method proj --trigger_epoch 20
    python scripts/train_symmetrization.py --method enforce --enforce_op proj
    python scripts/train_symmetrization.py --method orbit --orbit_lik prob
    python scripts/train_symmetrization.py --dataset NCT_CRC_HE --method gcnn --per_sample_backward
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import wandb

from ebnn.data import get_loaders, get_num_classes
from ebnn.histology import DEFAULT_IMAGE_SIZE
from ebnn.metrics import run_validation
from ebnn.models import BayesianFullyConvCNN, BayesianGroupConvCNN
from ebnn.orbit import ORBIT_LIKELIHOODS, OrbitAveragedBNN
from ebnn.p4cnn import GROUP_SIZE as P4_GROUP_SIZE
from ebnn.p4cnn import BayesianP4CNN
from ebnn.symmetrize import (
    apply_c4_enforcement,
    apply_one_shot_c4_posterior_average,
    apply_one_shot_c4_projection,
    apply_with_norm_rescale,
    build_optimizer,
    clear_optimizer_state,
    describe_optimizer,
    expand_small_model_to_gcnn,
    initialize_gaussian_posterior_scale,
    initialize_posterior_mean,
)


# --------------------------------------------------------------------------- #
# One training phase (shared by proj, avg, and both gcnn stages)
# --------------------------------------------------------------------------- #
def train_phase(
    model,
    train_loader,
    test_loader,
    equiv_loader,
    device,
    optimizer,
    *,
    num_epochs,
    beta,
    train_samples,
    eval_samples,
    val_interval,
    val_source,
    use_wandb,
    stage_tag=None,
    start_epoch=0,
    symmetrize_fn=None,
    symmetrize_label=None,
    trigger_acc=None,
    trigger_epoch=None,
    clear_state_on_symmetrize=True,
    post_step_fn=None,
    per_sample_backward=False,
):
    """Train ``model`` for ``num_epochs`` with periodic validation.

    ``post_step_fn(model)``, if given, runs after every optimiser step (used by
    ``enforce`` to re-symmetrise).  ``per_sample_backward`` backpropagates each MC
    sample separately: the same gradient as one backward of the averaged loss, but
    only one sample's graph is held in memory at a time.
    """
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None
    best_epoch = None
    best_metrics = None
    sym_applied = False

    def _validate(global_epoch, extra=None):
        nonlocal best_acc, best_state, best_epoch, best_metrics
        metrics, val_dist = run_validation(model, test_loader, equiv_loader, device, eval_samples, val_source)
        # avg / proj: only epochs after the symmetrisation count as "best".
        best_eligible = (symmetrize_fn is None) or sym_applied
        if best_eligible and metrics["val/accuracy"] >= best_acc:
            best_acc = metrics["val/accuracy"]
            best_epoch = global_epoch
            best_metrics = dict(metrics)
            best_state = copy.deepcopy(model.state_dict())
        log = {"epoch": global_epoch}
        if stage_tag is not None:
            log["stage"] = stage_tag
        log.update(metrics)
        if extra:
            log.update(extra)
        if use_wandb:
            log["val/pred_label_hist"] = wandb.Histogram(val_dist["pred_labels"].tolist())
            wandb.log(log)
        return metrics

    def _maybe_symmetrize(global_epoch, train_acc):
        nonlocal sym_applied
        if symmetrize_fn is None or sym_applied:
            return False
        acc_cond = (trigger_acc is not None) and (train_acc is not None) and (train_acc >= trigger_acc)
        epoch_cond = (trigger_epoch is not None) and (global_epoch >= trigger_epoch)
        if not (acc_cond or epoch_cond):
            return False
        symmetrize_fn(model)
        if clear_state_on_symmetrize:
            clear_optimizer_state(optimizer)
        sym_applied = True
        reasons = []
        if acc_cond:
            reasons.append(f"acc={train_acc:.2f}%>={trigger_acc:.2f}%")
        if epoch_cond:
            reasons.append(f"epoch={global_epoch}>={trigger_epoch}")
        reason = " and ".join(reasons) if reasons else "post-init"
        print(f"[{symmetrize_label}] applied at epoch {global_epoch} ({reason}); optimizer buffers cleared")
        if use_wandb:
            wandb.log(
                {
                    "epoch": global_epoch,
                    "symmetrize/event_epoch": global_epoch,
                    "symmetrize/op": symmetrize_label,
                    "symmetrize/trigger_acc": (train_acc if train_acc is not None else float("nan")),
                    "symmetrize/reason": reason,
                }
            )
        return True

    _maybe_symmetrize(global_epoch=start_epoch + 0, train_acc=None)
    _validate(start_epoch + 0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = total_nll = total_kl = 0.0
        total_correct = total_count = 0

        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            nll_sum = kl_sum = 0.0
            logits_last = None
            for _ in range(train_samples):
                logits, kl = model(inputs, return_kl=True)
                nll = criterion(logits, targets)
                if per_sample_backward:
                    ((nll + beta * kl) / train_samples).backward()
                    nll, kl = nll.detach(), kl.detach()
                nll_sum = nll_sum + nll
                kl_sum = kl_sum + kl
                logits_last = logits

            nll_avg = nll_sum / train_samples
            kl_avg = kl_sum / train_samples
            loss = nll_avg + beta * kl_avg
            if not per_sample_backward:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if post_step_fn is not None:
                post_step_fn(model)

            total_loss += loss.item() * inputs.size(0)
            total_nll += nll_avg.item() * inputs.size(0)
            total_kl += kl_avg.item() * inputs.size(0)
            total_correct += logits_last.argmax(dim=1).eq(targets).sum().item()
            total_count += targets.size(0)

        n = len(train_loader.dataset)
        train_acc = 100.0 * total_correct / total_count
        global_epoch = start_epoch + epoch + 1

        train_metrics = {
            "epoch": global_epoch,
            "train/loss": total_loss / n,
            "train/expected_nll": total_nll / n,
            "train/raw_kl": total_kl / n,
            "train/regularization_kl": beta * (total_kl / n),
            "train/accuracy": train_acc,
            "train/beta": beta,
            "symmetrize/applied": int(sym_applied),
        }
        if stage_tag is not None:
            train_metrics["stage"] = stage_tag

        triggered_now = _maybe_symmetrize(global_epoch, train_acc)
        if use_wandb:
            wandb.log(train_metrics)

        should_validate = (global_epoch % val_interval == 0) or (epoch + 1 == num_epochs)
        if triggered_now or should_validate:
            _validate(global_epoch)

    tag = stage_tag or "train"
    if best_metrics is not None:
        print(
            f"[{tag}] BEST val/acc={best_acc:.2f}% @ epoch {best_epoch} | "
            f"osp={best_metrics['val/osp']:.4f} "
            f"sym_kl={best_metrics['val/symmetric_kl_div_mean']:.4f} "
            f"val_loss={best_metrics['val/loss']:.4f} | "
            f"acc_c4={best_metrics['val/accuracy_c4']:.2f}% "
            f"acc_orig={best_metrics['val/accuracy_orig']:.2f}% (selected by {val_source})"
        )
        if use_wandb:
            prefix = f"best/{tag}/" if stage_tag is not None else "best/"
            summary = {
                prefix + "epoch": best_epoch,
                prefix + "val_accuracy": best_acc,
                prefix + "val_loss": best_metrics["val/loss"],
                prefix + "val_accuracy_c4": best_metrics["val/accuracy_c4"],
                prefix + "val_accuracy_orig": best_metrics["val/accuracy_orig"],
                prefix + "osp": best_metrics["val/osp"],
                prefix + "symmetric_kl_div_mean": best_metrics["val/symmetric_kl_div_mean"],
                prefix + "symmetric_kl_div_std": best_metrics["val/symmetric_kl_div_std"],
            }
            try:
                wandb.run.summary.update(summary)
            except Exception:
                wandb.log(summary)
    else:
        print(f"[{tag}] no eligible best checkpoint recorded")
    return {
        "best_acc": best_acc,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }


# --------------------------------------------------------------------------- #
# Per-method drivers
# --------------------------------------------------------------------------- #
def _build_fcnn(args, in_channels, device, conv_channels=None):
    model = BayesianFullyConvCNN(
        in_channels=in_channels,
        num_classes=args.num_classes,
        prior_params={"std": args.prior_std},
        conv_channels=args.conv_channels if conv_channels is None else conv_channels,
        rho_init=args.rho_init,
        random_prior=args.random_prior,
    ).to(device)
    initialize_posterior_mean(model, posterior_init=args.posterior_init)
    if args.posterior_init != "prior":
        initialize_gaussian_posterior_scale(model, rho_init=args.rho_init)
    return model


def _make_optimizer(model, args):
    opt = build_optimizer(
        model,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.nesterov,
        rho_lr_mult=args.rho_lr_mult,
        rho_lr=args.rho_lr,
        rho_weight_decay=args.rho_weight_decay,
    )
    print("Optimizer:", describe_optimizer(opt))
    return opt


def run_avg_or_proj(args, loaders, in_channels, device):
    train_loader, test_loader, equiv_loader = loaders
    model = _build_fcnn(args, in_channels, device)
    optimizer = _make_optimizer(model, args)
    beta = (1.0 / len(train_loader.dataset)) if args.beta is None else args.beta

    post_step_fn = None
    if args.method == "baseline":
        symmetrize_fn = None
        label = "baseline"
    elif args.method == "proj":
        symmetrize_fn = apply_one_shot_c4_projection
        label = "proj"
    elif args.method == "enforce":
        label = "enforce/" + ("proj" if args.enforce_op == "proj" else f"avg/{args.avg_method}")

        def symmetrize_fn(m):
            factors = apply_c4_enforcement(
                m,
                op=args.enforce_op,
                avg_method=args.avg_method,
                rescale=args.enforce_rescale in ("init", "always"),
            )
            if factors:
                print(
                    f"[{label}] init rescale factors (mu, sigma): "
                    + ", ".join(f"{name} ({mu:.3f}, {sig:.3f})" for name, (mu, sig) in factors.items())
                )

        def post_step_fn(m):
            apply_c4_enforcement(
                m, op=args.enforce_op, avg_method=args.avg_method, rescale=args.enforce_rescale == "always"
            )

    else:

        def symmetrize_fn(m):
            return apply_one_shot_c4_posterior_average(m, method=args.avg_method)

        label = f"avg/{args.avg_method}"

    if args.sym_rescale and args.method in ("avg", "proj"):
        one_shot_fn = symmetrize_fn
        label = label + "+rescale"

        def symmetrize_fn(m):
            factors = apply_with_norm_rescale(m, one_shot_fn)
            print(
                f"[{label}] rescale factors (mu, sigma): "
                + ", ".join(f"{name} ({mu:.3f}, {sig:.3f})" for name, (mu, sig) in factors.items())
            )

    result = train_phase(
        model,
        train_loader,
        test_loader,
        equiv_loader,
        device,
        optimizer,
        num_epochs=args.epochs,
        beta=beta,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        val_interval=args.val_interval,
        val_source=args.val_source,
        use_wandb=not args.no_wandb,
        symmetrize_fn=symmetrize_fn,
        symmetrize_label=label,
        trigger_acc=args.trigger_acc,
        trigger_epoch=args.trigger_epoch,
        clear_state_on_symmetrize=not args.no_clear_state,
        post_step_fn=post_step_fn,
        per_sample_backward=args.per_sample_backward,
    )
    if result["best_state"] is not None:
        model.load_state_dict(result["best_state"])
    return model


def run_gcnn(args, loaders, in_channels, device):
    train_loader, test_loader, equiv_loader = loaders
    beta = (1.0 / len(train_loader.dataset)) if args.beta is None else args.beta

    base_channels = [c // args.group_size for c in args.conv_channels]
    print("\n" + "=" * 60 + "\nSTAGE 1: small baseline model\n" + "=" * 60)
    print(f"[gcnn] stage-1 channels {base_channels} -> stage-2 channels {args.conv_channels}")
    small_model = _build_fcnn(args, in_channels, device, conv_channels=base_channels)
    if args.stage1_epochs == 0:
        best_small_state = copy.deepcopy(small_model.state_dict())
        print("Stage 1 skipped (stage1_epochs=0).")
    else:
        opt1 = _make_optimizer(small_model, args)
        stage1_result = train_phase(
            small_model,
            train_loader,
            test_loader,
            equiv_loader,
            device,
            opt1,
            num_epochs=args.stage1_epochs,
            beta=beta,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            val_interval=args.val_interval,
            val_source=args.val_source,
            use_wandb=not args.no_wandb,
            stage_tag="stage1",
            per_sample_backward=args.per_sample_backward,
        )
        best_small_state = stage1_result["best_state"]
    if best_small_state is None:
        best_small_state = copy.deepcopy(small_model.state_dict())
    small_model.load_state_dict(best_small_state)

    print("\n" + "=" * 60 + "\nSTAGE 2: expand to GCNN and train\n" + "=" * 60)
    gcnn_model = BayesianGroupConvCNN(
        in_channels=in_channels,
        num_classes=args.num_classes,
        prior_params={"std": args.prior_std},
        base_channels=base_channels,
        group_size=args.group_size,
        rho_init=args.rho_init,
        random_prior=args.random_prior,
    ).to(device)
    expand_small_model_to_gcnn(
        small_model, gcnn_model, group_size=args.group_size, arrangement=args.arrangement
    )
    print(f"Initialized GCNN from small model via C{args.group_size} expansion ({args.arrangement}).")

    opt2 = _make_optimizer(gcnn_model, args)  # fresh buffers => no stale-state issue
    stage2_result = train_phase(
        gcnn_model,
        train_loader,
        test_loader,
        equiv_loader,
        device,
        opt2,
        num_epochs=args.stage2_epochs,
        beta=beta,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        val_interval=args.val_interval,
        val_source=args.val_source,
        use_wandb=not args.no_wandb,
        stage_tag="stage2",
        start_epoch=args.stage1_epochs,
        per_sample_backward=args.per_sample_backward,
    )
    if stage2_result["best_state"] is not None:
        gcnn_model.load_state_dict(stage2_result["best_state"])
    return gcnn_model


def run_p4cnn(args, loaders, in_channels, device):
    """Bayesian C4 G-CNN trained from scratch for ``--epochs`` (no symmetrisation step needed)."""
    train_loader, test_loader, equiv_loader = loaders
    beta = (1.0 / len(train_loader.dataset)) if args.beta is None else args.beta

    fields = [c // P4_GROUP_SIZE for c in args.conv_channels]  # 4 rotations per field
    model = BayesianP4CNN(
        in_channels=in_channels,
        num_classes=args.num_classes,
        prior_params={"std": args.prior_std},
        base_channels=fields,
        rho_init=args.rho_init,
        random_prior=args.random_prior,
    ).to(device)
    initialize_posterior_mean(model, posterior_init=args.posterior_init)
    if args.posterior_init != "prior":
        initialize_gaussian_posterior_scale(model, rho_init=args.rho_init)
    n_free = sum(p.numel() for name, p in model.named_parameters() if name.endswith("_mu"))
    print(
        f"P4CNN channels {args.conv_channels} = fields {fields} x 4 rotations: {n_free} free posterior means"
    )

    optimizer = _make_optimizer(model, args)
    result = train_phase(
        model,
        train_loader,
        test_loader,
        equiv_loader,
        device,
        optimizer,
        num_epochs=args.epochs,
        beta=beta,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        val_interval=args.val_interval,
        val_source=args.val_source,
        use_wandb=not args.no_wandb,
        per_sample_backward=args.per_sample_backward,
    )
    if result["best_state"] is not None:
        model.load_state_dict(result["best_state"])
    return model


def run_orbit(args, loaders, in_channels, device):
    """Orbit-averaged likelihood (Nabarro et al., 2022); everything but the likelihood as for avg / proj."""
    train_loader, test_loader, equiv_loader = loaders
    beta = (1.0 / len(train_loader.dataset)) if args.beta is None else args.beta

    base = _build_fcnn(args, in_channels, device)
    model = OrbitAveragedBNN(base, likelihood=args.orbit_lik).to(device)
    print(
        f"Orbit-averaged likelihood ({args.orbit_lik}) on BayesianFullyConvCNN {args.conv_channels}: "
        f"one posterior sample per C4 orbit, 4x forward cost"
    )

    optimizer = _make_optimizer(model, args)
    result = train_phase(
        model,
        train_loader,
        test_loader,
        equiv_loader,
        device,
        optimizer,
        num_epochs=args.epochs,
        beta=beta,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        val_interval=args.val_interval,
        val_source=args.val_source,
        use_wandb=not args.no_wandb,
        per_sample_backward=args.per_sample_backward,
    )
    if result["best_state"] is not None:
        model.load_state_dict(result["best_state"])
    return model


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description="Unified C4-equivariant Bayesian CNN trainer")

    p.add_argument(
        "--method",
        type=str,
        default="gcnn",
        choices=["avg", "proj", "gcnn", "baseline", "enforce", "p4cnn", "orbit"],
    )
    p.add_argument("--optimizer", type=str, default="sgd", choices=["adamw", "sgd"])

    p.add_argument(
        "--dataset",
        type=str,
        default="FashionMNIST",
        choices=["MNIST", "FashionMNIST", "CIFAR10", "NCT_CRC_HE"],
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=None,
        help=f"NCT_CRC_HE only: patch side length (default {DEFAULT_IMAGE_SIZE} = native).",
    )
    p.add_argument("--train_size", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--conv_channels",
        type=int,
        nargs="+",
        default=None,
        help="Feature-map channels per conv layer, for every method (gcnn: stage-2 width, its stage-1 "
        "model has conv_channels // group_size; p4cnn: conv_channels // 4 fields x 4 rotations).",
    )
    p.add_argument("--prior_std", type=float, default=1.0)
    p.add_argument("--rho_init", type=float, default=-3.0)
    p.add_argument("--random_prior", action="store_true")
    p.add_argument("--posterior_init", type=str, default="kaiming", choices=["zero", "kaiming", "prior"])
    p.add_argument("--augmentation", type=str, default="full", choices=["without", "full", "random"])

    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay on the non-rho group. Default 0.0 (canonical SGD run).",
    )
    p.add_argument("--momentum", type=float, default=0.9, help="SGD only.")
    p.add_argument(
        "--nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SGD Nesterov momentum (use --no-nesterov to disable).",
    )
    p.add_argument("--rho_lr_mult", type=float, default=10.0, help="SGD: rho group LR = lr * this.")
    p.add_argument("--rho_lr", type=float, default=None, help="SGD: explicit rho group LR.")
    p.add_argument("--rho_weight_decay", type=float, default=0.0, help="SGD: weight decay on rho group.")

    p.add_argument("--beta", type=float, default=None, help="KL weight (default 1/N).")
    p.add_argument("--train_samples", type=int, default=10)
    p.add_argument(
        "--per_sample_backward",
        action="store_true",
        help="Backpropagate each MC training sample separately: same gradient, ~train_samples x less "
        "activation memory (needed for the GCNN at 224 px).",
    )
    p.add_argument("--eval_samples", type=int, default=16)
    p.add_argument("--epochs", type=int, default=500, help="avg/proj/baseline/enforce/p4cnn/orbit epochs.")
    p.add_argument("--stage1_epochs", type=int, default=100, help="gcnn stage-1 epochs.")
    p.add_argument("--stage2_epochs", type=int, default=400, help="gcnn stage-2 epochs.")
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument(
        "--val_source",
        type=str,
        default="equiv_all",
        choices=["test", "equiv_all"],
        help=(
            "Source for val/accuracy+loss and the best checkpoint: equiv_all = C4-augmented test set "
            "(all 4 rotations), test = original test images. Both are always logged "
            "(val/accuracy_c4, val/accuracy_orig); osp/sym_kl always use the full orbit."
        ),
    )

    trig = p.add_mutually_exclusive_group()
    trig.add_argument(
        "--trigger_acc", type=float, default=None, help="Symmetrize when train acc >= this (avg/proj)."
    )
    trig.add_argument(
        "--trigger_epoch",
        type=int,
        default=None,
        help="Symmetrize when epoch >= this; 0 = after init (avg/proj).",
    )
    p.add_argument("--avg_method", type=str, default="geometric", choices=["geometric", "arithmetic"])
    p.add_argument(
        "--orbit_lik",
        type=str,
        default="prob",
        choices=list(ORBIT_LIKELIHOODS),
        help="orbit (Nabarro et al., 2022): average the 4 rotations' softmax probabilities (prob) "
        "or logits (logit) under one shared posterior sample.",
    )
    p.add_argument(
        "--sym_rescale",
        action="store_true",
        help="avg / proj: after the one-shot symmetrisation, rescale each weight tensor (posterior "
        "mean and std) back to its pre-symmetrisation norm (as --enforce_rescale init), then train "
        "unconstrained.",
    )
    p.add_argument(
        "--no_clear_state", action="store_true", help="Do NOT clear optimizer buffers after symmetrization."
    )
    p.add_argument(
        "--enforce_op",
        type=str,
        default="proj",
        choices=["proj", "avg"],
        help="enforce: operator applied at init and after every optimizer step (avg uses --avg_method).",
    )
    p.add_argument(
        "--enforce_rescale",
        type=str,
        default="init",
        choices=["init", "always", "none"],
        help="enforce: rescale each weight tensor (posterior mean and std) back to its "
        "pre-symmetrisation norm at init only / after every application / never.",
    )

    p.add_argument("--group_size", type=int, default=4)
    p.add_argument(
        "--arrangement",
        type=str,
        default="gcnn",
        choices=["gcnn", "row_rolled", "row_constant"],
        help="Orbit-expansion filter tiling (only used when --method gcnn). "
        "gcnn = main results; row_rolled / row_constant = ablation variants.",
    )

    p.add_argument("--project", type=str, default="training_all_in_one")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--save_path", type=str, default=None)
    return p


def main():
    import os

    args = build_arg_parser().parse_args()

    if args.method in ("avg", "proj") and args.trigger_acc is None and args.trigger_epoch is None:
        args.trigger_acc = 50.0
    if args.sym_rescale and args.method not in ("avg", "proj"):
        raise SystemExit(f"--sym_rescale only applies to --method avg / proj (got {args.method})")
    if args.method == "enforce":
        if args.trigger_acc is not None or args.trigger_epoch not in (None, 0):
            raise SystemExit("--method enforce symmetrises at init; drop --trigger_acc / --trigger_epoch")
        args.trigger_epoch = 0
    if args.trigger_epoch is not None and args.trigger_epoch < 0:
        raise SystemExit("--trigger_epoch must be >= 0")

    if args.dataset == "NCT_CRC_HE":
        if args.image_size is None:
            args.image_size = DEFAULT_IMAGE_SIZE
    elif args.image_size is not None:
        raise SystemExit("--image_size is only supported for --dataset NCT_CRC_HE")
    args.num_classes = get_num_classes(args.dataset)

    if args.method == "gcnn":
        args.epochs = args.stage1_epochs + args.stage2_epochs
        print(
            f"[gcnn] total epochs = stage1({args.stage1_epochs}) + stage2({args.stage2_epochs}) = {args.epochs}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | method={args.method} | optimizer={args.optimizer}")

    train_loader, test_loader, equiv_loader = get_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train_size=args.train_size,
        augmentation_mode=args.augmentation,
        image_size=args.image_size,
    )
    sample_inputs = next(iter(train_loader))[0]
    in_channels = sample_inputs.shape[1]

    if args.conv_channels is None:
        if args.dataset == "NCT_CRC_HE":
            # five conv + 2x2-pool blocks: 224 -> 7
            args.conv_channels = [32, 64, 128, 256, 256]
        else:
            args.conv_channels = [64, 128, 256] if args.dataset == "CIFAR10" else [32, 64]
    group = {"gcnn": args.group_size, "p4cnn": P4_GROUP_SIZE}.get(args.method)
    if group is not None and any(c % group != 0 for c in args.conv_channels):
        raise SystemExit(
            f"--method {args.method}: every --conv_channels entry must be divisible by {group} "
            f"(got {args.conv_channels})"
        )
    side = sample_inputs.shape[-1]
    if side % (2 ** len(args.conv_channels)) != 0:
        print(
            f"Warning: input size {side} is not divisible by 2**{len(args.conv_channels)}; max-pooling an "
            "odd-sized feature map breaks exact C4 equivariance."
        )

    method_tag = {"enforce": f"enforce-{args.enforce_op}", "orbit": f"orbit-{args.orbit_lik}"}.get(
        args.method, args.method
    )
    if not args.no_wandb:
        if args.run_name is None:
            args.run_name = f"{method_tag}_{args.optimizer}_{args.dataset}_{args.augmentation}_{args.seed}"
        wandb.init(project=args.project, name=args.run_name, config=vars(args))

    loaders = (train_loader, test_loader, equiv_loader)
    if args.method in ("avg", "proj", "baseline", "enforce"):
        model = run_avg_or_proj(args, loaders, in_channels, device)
    elif args.method == "p4cnn":
        model = run_p4cnn(args, loaders, in_channels, device)
    elif args.method == "orbit":
        model = run_orbit(args, loaders, in_channels, device)
    else:
        model = run_gcnn(args, loaders, in_channels, device)

    if args.save_model:
        if args.save_path is None:
            name = f"{method_tag}_{args.optimizer}_{args.dataset}_{args.augmentation}_{args.seed}.pt"
            args.save_path = os.path.join("trained_models", name)
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, args.save_path)
        print(f"Saved model checkpoint to: {args.save_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
