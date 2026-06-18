"""Train a configurable Bayesian CNN with a selectable variational family
(Gaussian / Laplace / Log-Normal) on C4-augmented data.

This backs the appendix on closedness of the exponential family under the group
action: the mean-field Gaussian family is closed under the C4 push-forward (so
the equivariance results apply), whereas Laplace / Log-Normal are not, which is
reflected in their weaker equivariance metrics.

Supports KL annealing (separate conv / fc schedules), free bits, and reports
test accuracy plus equivariance metrics (orbit consistency + symmetric KL) on
both the held-out and training C4-orbit sets.

Example:
    python scripts/train_multifamily.py --dataset FashionMNIST --family gaussian \\
        --train_size 5000 --epochs 100 --seed 42
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from ebnn.data import EquivarianceEvalSet, get_loaders
from ebnn.metrics import compute_orbits_same_pred, compute_symmetric_kl_divergence
from ebnn.models import BayesianCNNConfigurable


def train_bayesian_cnn(
    model,
    train_loader,
    test_loader,
    equiv_loader,
    device,
    num_epochs=100,
    learning_rate=1e-3,
    kl_weight=None,
    kl_weight_fc=None,
    train_samples=10,
    eval_samples=10,
    val_interval=10,
    use_wandb=True,
    run_name=None,
    kl_annealing_epochs=0,
    kl_annealing_epochs_fc=0,
    free_bits=0.0,
    weight_decay=1e-4,
):
    """Train a Bayesian CNN on rotation-augmented data with equivariance evaluation.

    KL annealing (when enabled) uses a multi-stage schedule: zero weight during
    a warmup of ``kl_annealing_epochs`` epochs, then a linear ramp to the target
    weight up to epoch 100, then fixed at the target.  Conv and FC layers have
    independent schedules.
    """
    target_kl_weight_conv = 1.0 / len(train_loader.dataset) if kl_weight is None else kl_weight
    target_kl_weight_fc = target_kl_weight_conv if kl_weight_fc is None else kl_weight_fc

    current_kl_weight_conv = 0.0 if kl_annealing_epochs > 0 else target_kl_weight_conv
    current_kl_weight_fc = 0.0 if kl_annealing_epochs_fc > 0 else target_kl_weight_fc

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(num_epochs):
        # KL annealing (conv)
        if kl_annealing_epochs > 0:
            warmup_end, annealing_end = kl_annealing_epochs, 100
            if epoch < warmup_end:
                current_kl_weight_conv = 0.0
            elif epoch < annealing_end:
                progress = (epoch - warmup_end) / (annealing_end - warmup_end)
                current_kl_weight_conv = target_kl_weight_conv * progress
            else:
                current_kl_weight_conv = target_kl_weight_conv

        # KL annealing (fc, independent schedule)
        if kl_annealing_epochs_fc > 0:
            warmup_end_fc, annealing_end_fc = kl_annealing_epochs_fc, 100
            if epoch < warmup_end_fc:
                current_kl_weight_fc = 0.0
            elif epoch < annealing_end_fc:
                progress_fc = (epoch - warmup_end_fc) / (annealing_end_fc - warmup_end_fc)
                current_kl_weight_fc = target_kl_weight_fc * progress_fc
            else:
                current_kl_weight_fc = target_kl_weight_fc

        model.train()
        total_train_loss = 0.0
        total_nll_loss = 0.0
        total_kl_loss = 0.0
        total_raw_kl_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            if train_samples == 1:
                outputs, conv_kl_raw, fc_kl_raw = model(inputs, return_kl=True, return_separate_kl=True)
                nll_loss = criterion(outputs, targets)
            else:
                total_nll = 0.0
                total_conv_kl = 0.0
                total_fc_kl = 0.0
                for _ in range(train_samples):
                    outputs, conv_kl_s, fc_kl_s = model(inputs, return_kl=True, return_separate_kl=True)
                    total_nll += criterion(outputs, targets)
                    total_conv_kl += conv_kl_s
                    total_fc_kl += fc_kl_s
                nll_loss = total_nll
                conv_kl_raw = total_conv_kl
                fc_kl_raw = total_fc_kl

            kl_loss_raw = conv_kl_raw + fc_kl_raw

            if free_bits > 0:
                num_kl_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                min_kl = free_bits * num_kl_params
                conv_kl = torch.clamp(conv_kl_raw, min=min_kl * 0.5)
                fc_kl = torch.clamp(fc_kl_raw, min=min_kl * 0.5)
            else:
                conv_kl = conv_kl_raw
                fc_kl = fc_kl_raw

            kl_loss = conv_kl + fc_kl
            loss = (
                nll_loss + current_kl_weight_conv * conv_kl + current_kl_weight_fc * fc_kl
            ) / train_samples

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * inputs.size(0)
            total_nll_loss += nll_loss.item() * inputs.size(0)
            total_kl_loss += kl_loss.item() * inputs.size(0)
            total_raw_kl_loss += kl_loss_raw.item() * inputs.size(0)
            correct += outputs.max(1)[1].eq(targets).sum().item()
            total += targets.size(0)

        n = len(train_loader.dataset)
        scheduler.step()

        train_metrics = {
            "epoch": epoch + 1,
            "train/loss": total_train_loss / n,
            "train/nll": total_nll_loss / n,
            "train/kl": total_kl_loss / n,
            "train/kl_raw": total_raw_kl_loss / n,
            "train/kl_weight_conv": current_kl_weight_conv,
            "train/kl_weight_fc": current_kl_weight_fc,
            "train/accuracy": 100.0 * correct / total,
            "train/lr": scheduler.get_last_lr()[0],
        }
        if use_wandb:
            wandb.log(train_metrics)

        if (epoch + 1) % val_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
            model.eval()
            total_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    all_outputs = [model(inputs, return_kl=False) for _ in range(eval_samples)]
                    avg_outputs = torch.stack(all_outputs).mean(dim=0)
                    total_val_loss += criterion(avg_outputs, targets).item() * inputs.size(0)
                    correct_val += avg_outputs.max(1)[1].eq(targets).sum().item()
                    total_val += targets.size(0)
            avg_val_loss = total_val_loss / len(test_loader.dataset)
            val_acc = 100.0 * correct_val / total_val

            # Equivariance metrics on the held-out set.
            kl_test = compute_symmetric_kl_divergence(model, device, equiv_loader)
            osp_test = compute_orbits_same_pred(model, device, equiv_loader)

            # Equivariance metrics on a C4-orbit set built from the training data.
            train_base = getattr(train_loader.dataset, "base_dataset", train_loader.dataset)
            train_equiv_dataset = EquivarianceEvalSet(train_base, n_samples=min(2000, len(train_base)))
            train_equiv_loader = DataLoader(
                train_equiv_dataset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
            )
            kl_train = compute_symmetric_kl_divergence(model, device, train_equiv_loader)
            osp_train = compute_orbits_same_pred(model, device, train_equiv_loader)

            val_metrics = {
                "epoch": epoch + 1,
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc,
                "equivariance_test/orbits_same_pred": osp_test,
                "equivariance_test/symmetric_kl_div_mean": kl_test["mean"],
                "equivariance_test/symmetric_kl_div_std": kl_test["std"],
                "equivariance_train/orbits_same_pred": osp_train,
                "equivariance_train/symmetric_kl_div_mean": kl_train["mean"],
                "equivariance_train/symmetric_kl_div_std": kl_train["std"],
            }
            if use_wandb:
                wandb.log(val_metrics)

            if val_acc > best_acc:
                best_acc = val_acc
                if use_wandb:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_acc": val_acc,
                            "metrics": {**train_metrics, **val_metrics},
                        },
                        f"{run_name}.pt",
                    )

    print(f"\nTraining completed. Best validation accuracy: {best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Train a configurable Bayesian CNN on rotation-augmented data"
    )
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "CIFAR10"])
    parser.add_argument(
        "--train_size", type=int, default=None, help="Number of training samples (None = use all)"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--family",
        type=str,
        default="gaussian",
        choices=["gaussian", "laplace", "lognormal"],
        help="Exponential family for the variational posterior",
    )
    parser.add_argument(
        "--conv_channels",
        type=int,
        nargs="+",
        default=None,
        help="Conv channel sizes (default: [64, 128, 256] for CIFAR, [64, 128] for MNIST)",
    )
    parser.add_argument("--prior_std", type=float, default=1.0, help="Prior std (Gaussian)")
    parser.add_argument("--prior_scale", type=float, default=1.0, help="Prior scale (Laplace)")
    parser.add_argument(
        "--rho_init", type=float, default=-3.0, help="Initial rho for the Gaussian/LogNormal scale"
    )
    parser.add_argument("--random_prior", action="store_true", help="Use a random prior mean (Gaussian only)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--kl_weight", type=float, default=None, help="KL weight for conv layers (1/N if None)"
    )
    parser.add_argument(
        "--kl_weight_fc", type=float, default=None, help="KL weight for fc layers (= kl_weight if None)"
    )
    parser.add_argument(
        "--kl_annealing_epochs", type=int, default=0, help="Warmup epochs for conv KL annealing"
    )
    parser.add_argument(
        "--kl_annealing_epochs_fc", type=int, default=0, help="Warmup epochs for fc KL annealing"
    )
    parser.add_argument(
        "--free_bits", type=float, default=0.0, help="Minimum KL per latent dimension (0 = off)"
    )
    parser.add_argument("--train_samples", type=int, default=10, help="MC samples during training")
    parser.add_argument("--eval_samples", type=int, default=10, help="MC samples during evaluation")
    parser.add_argument("--val_interval", type=int, default=10, help="Validate every N epochs")
    parser.add_argument("--wandb_watch", action="store_true", help="Enable wandb.watch for gradients")
    parser.add_argument("--project", type=str, default="bayesian-cnn-augmented")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_augmentation", action="store_true", help="Disable C4 augmentation (baseline comparison)"
    )
    args = parser.parse_args()

    print("=" * 30)
    print("Running with arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 30)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_augmentation = not args.no_augmentation
    print(
        f"Loading {args.dataset} {'with C4 augmentation' if use_augmentation else 'WITHOUT augmentation'}..."
    )
    train_loader, test_loader, equiv_loader = get_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train_size=args.train_size,
        use_augmentation=use_augmentation,
    )
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    sample_input = next(iter(train_loader))[0]
    in_channels = sample_input.shape[1]
    input_size = (int(sample_input.shape[2]), int(sample_input.shape[3]))
    num_classes = 10

    prior_params = (
        {"std": args.prior_std}
        if args.family == "gaussian"
        else {"scale": args.prior_scale} if args.family == "laplace" else {"mu": 0.0, "sigma": args.prior_std}
    )

    if args.conv_channels is None:
        args.conv_channels = [64, 128, 256] if args.dataset in ("CIFAR10", "CIFAR100") else [64, 128]

    model = BayesianCNNConfigurable(
        in_channels=in_channels,
        num_classes=num_classes,
        family=args.family,
        prior_params=prior_params,
        conv_channels=args.conv_channels,
        input_size=input_size,
        rho_init=args.rho_init,
        random_prior=args.random_prior,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    use_wandb = not args.no_wandb
    if use_wandb:
        if args.run_name is None:
            aug_suffix = "woaug" if args.no_augmentation else "aug"
            args.run_name = f"cnn_{args.family}_{args.dataset}_{aug_suffix}_{args.seed}"
        wandb.init(project=args.project, name=args.run_name, config=vars(args))
        if args.wandb_watch:
            wandb.watch(model, log="gradients", log_freq=200)

    train_bayesian_cnn(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        equiv_loader=equiv_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
        kl_weight_fc=args.kl_weight_fc,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        val_interval=args.val_interval,
        use_wandb=use_wandb,
        run_name=args.run_name,
        kl_annealing_epochs=args.kl_annealing_epochs,
        kl_annealing_epochs_fc=args.kl_annealing_epochs_fc,
        free_bits=args.free_bits,
        weight_decay=args.weight_decay,
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
