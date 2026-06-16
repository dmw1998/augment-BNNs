"""Theorem-3 verification: train a baseline Bayesian CNN and measure the empirical equivariance defect on a held-out C4-orbit evaluation set.

Example:
    python scripts/train_theorem3.py --dataset FashionMNIST --train_size 5000 --epochs 500 --eval_samples 10 --seed 42
"""

import argparse

import numpy as np
import torch
import wandb

from ebnn.data import get_loaders
from ebnn.metrics import compute_equivariance_defect
from ebnn.models import BayesianCNN
from ebnn.training import evaluate_mc_accuracy, train_baseline_bnn


def main():
    parser = argparse.ArgumentParser(description="Theorem 3 verification: train BNN with different priors")
    parser.add_argument(
        "--dataset", type=str, default="FashionMNIST", choices=["MNIST", "FashionMNIST", "CIFAR10"]
    )
    parser.add_argument(
        "--train_size",
        type=int,
        required=True,
        help="N_0: number of training samples (before C4 augmentation)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--conv_channels", type=int, nargs="+", default=None)
    parser.add_argument("--prior_std", type=float, default=1.0)
    parser.add_argument("--rho_init", type=float, default=-3.0)
    parser.add_argument("--random_prior", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--train_samples", type=int, default=10)
    parser.add_argument("--eval_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="theorem3_verification")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 50)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, equiv_loader = get_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train_size=args.train_size,
        use_augmentation=True,
    )
    print(
        f"Train samples (after aug): {len(train_loader.dataset)}, "
        f"Test: {len(test_loader.dataset)}, Equiv: {len(equiv_loader.dataset)}"
    )

    sample_input = next(iter(train_loader))[0]
    in_channels = sample_input.shape[1]
    input_size = (sample_input.shape[2], sample_input.shape[3])

    if args.conv_channels is None:
        args.conv_channels = [64, 128] if args.dataset == "CIFAR10" else [32, 64]

    model = BayesianCNN(
        in_channels=in_channels,
        num_classes=10,
        conv_channels=args.conv_channels,
        input_size=input_size,
        prior_std=args.prior_std,
        rho_init=args.rho_init,
        random_prior=args.random_prior,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    use_wandb = not args.no_wandb
    if use_wandb:
        if args.run_name is None:
            prior_tag = "rndprior" if args.random_prior else "isoprior"
            args.run_name = f"{args.dataset}_n{args.train_size}_{prior_tag}_seed{args.seed}"
        wandb.init(project=args.project, name=args.run_name, config=vars(args))

    train_baseline_bnn(
        model=model,
        train_loader=train_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        train_samples=args.train_samples,
        weight_decay=args.weight_decay,
        use_wandb=use_wandb,
    )

    test_acc = evaluate_mc_accuracy(model, test_loader, device, mc_samples=args.eval_samples)
    print(f"Test accuracy: {test_acc:.2f}%")

    eq_defect = compute_equivariance_defect(model, device, equiv_loader, mc_samples=args.eval_samples)
    print(f"Equivariance defect: mean={eq_defect['mean']:.6f}, std={eq_defect['std']:.6f}")

    if use_wandb:
        wandb.log(
            {
                "final/test_accuracy": test_acc,
                "final/equivariance_defect_mean": eq_defect["mean"],
                "final/equivariance_defect_std": eq_defect["std"],
                "final/train_size": args.train_size,
                "final/random_prior": args.random_prior,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
