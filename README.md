# Equivariance and Augmentation for Bayesian Neural Networks

Reference implementation for the experiments in *"Equivariance and Augmentation for Bayesian Neural Networks"*. The code studies whether **data augmentation alone** can induce **C₄ (90° rotation) equivariance** in variational Bayesian neural networks, and provides three mechanisms for turning an augmentation-trained posterior into an equivariant one:

- **`avg`** — one-shot C₄ posterior averaging over the orbit (geometric averaging or arithmetic moment matching).
- **`proj`** — one-shot projection of the posterior mean and scale onto the C₄-equivariant subspace.
- **`gcnn`** — block-circulant expansion of a trained small model into a group-convolutional network.

It also contains the empirical verification scripts for the sample-complexity results (the equivariance defect $Δ_{F}^{eq}$ and its Monte-Carlo behaviour as a function of the number of posterior samples $T$).

## Repository layout

```
ebnn/                     # importable package
├── layers.py             # Bayesian layers (Gaussian / Laplace / Log-Normal)
├── models.py             # BayesianCNN, configurable CNN, fully-conv & group-conv CNNs
├── data.py               # C4 augmentation datasets + get_loaders
├── metrics.py            # equivariance defect, orbit consistency, symmetric KL (MC + non-MC)
├── symmetrize.py         # avg / proj / gcnn operators, optimiser + re-init helpers
└── training.py           # shared ELBO training + MC evaluation loops
scripts/
├── train_symmetrization.py  # main trainer: --method {avg, proj, gcnn}
├── train_theorem3.py        # random prior + invariant prior
└── train_theorem4.py        # T-sweep + K-fold MC defect (sample complexity)
tests/
└── test_smoke.py            # CPU smoke tests (no dataset / GPU required)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `ebnn` package (editable) and its dependencies
(`torch`, `torchvision`, `numpy`, `wandb`). A CUDA-enabled PyTorch build is recommended for training; the code falls back to CPU automatically.

Datasets (MNIST / FashionMNIST / CIFAR-10) are downloaded on first use via `torchvision` into `./data`. The C₄-augmented and equivariance-evaluation sets are precomputed and cached under `./data/precomputed_aug`.

## Reproducing the experiments

All training runs log to [Weights & Biases](https://wandb.ai). Pass `--no_wandb` to disable logging, or set the project with `--project`. The main results are averaged over 5 seeds; the example commands below use a single seed.

### Inducing equivariance (main results)

The strongest configuration in the paper is orbit expansion (`gcnn`) trained with SGD on FashionMNIST (N₀ = 5000, full C₄ augmentation). **This is the default configuration**, so the bare command reproduces it (gcnn + SGD, `conv_channels [32, 64]`, 100 + 400 stage epochs, `lr 1e-2`, `weight_decay 0`, Nesterov, `seed 13`):

```bash
python scripts/train_symmetrization.py            # canonical run, seed 13
python scripts/train_symmetrization.py --seed 7   # same config, different seed (for the 5-seed average)
```

Posterior averaging and projection (applied mid-training once training accuracy crosses `--trigger_acc`, default 50%):

```bash
# C4 posterior averaging (geometric product-of-experts)
python scripts/train_symmetrization.py --method avg --avg_method geometric

# Projection onto the C4-equivariant subspace
python scripts/train_symmetrization.py --method proj
```

Useful options: `--optimizer {adamw,sgd}`, `--nesterov/--no-nesterov`, `--trigger_epoch N` (symmetrise at a fixed epoch instead of an accuracy threshold; `0` = right after init), `--posterior_init {zero,kaiming,prior}`, `--augmentation {without,full,random}`, `--conv_channels 32 64`, `--save_model`.

> **Why SGD for `gcnn`/symmetrisation.** SGD's linear update commutes with the group action, whereas Adam's per-parameter momentum does not; clearing the optimiser buffers at symmetrisation (the default) avoids stale non-equivariant momentum leaking through. Disable with `--no_clear_state`.

### Priors for equivariance defect (Theorem-3)

To train with a random prior, use the `--random_prior` flag (default: False, i.e. invariant prior):
```bash
python scripts/train_theorem3.py --dataset FashionMNIST --train_size 5000 \
    --epochs 500 --eval_samples 10 --seed 5 --random_prior
```

```bash
python scripts/train_theorem3.py --dataset FashionMNIST --train_size 5000 \
    --epochs 500 --eval_samples 10 --seed 5
```

### Monte-Carlo sample complexity (Theorem-4)

Sweeps the equivariance defect over T (powers of two up to `--eval_samples`) and runs a K-fold MC re-evaluation to estimate the per-realisation deviation std, predicted to scale as $O(1/\sqrt{T})$:

```bash
python scripts/train_theorem4.py --dataset FashionMNIST --train_size 5000 \
    --epochs 500 --eval_samples 1024 --K_mc_runs 10 --skip_single_sweep
```

## Tests

```bash
python tests/test_smoke.py        # or: python -m pytest tests/test_smoke.py
```

The smoke tests run on CPU without any dataset: they build every model, run a forward/backward pass, verify that the C₄ projection is idempotent, exercise the GCNN expansion, and check the optimiser builders.

## Notes on reproducibility

- **Posterior initialisation regimes.** Two initialisations are used and are preserved exactly via explicit layer arguments. The Theorem-3/4 baselines use Kaiming-initialised posterior means with `softplus(rho_init)` scale; the fully-/group-convolutional models use zero means with `softplus(0.693) ≈ 1` scale (and the unified trainer additionally re-initialises means/scales at startup via `initialize_posterior_mean` / `initialize_gaussian_posterior_scale`).
- **Functional, not bit-exact.** This is a cleaned/restructured version of the original research code. It is numerically equivalent to the original (same architectures, KL terms, ELBO, and training math), so re-running reproduces results up to the usual seed and hardware (CUDA) non-determinism. It is **not** guaranteed to be bit-identical to the original checkpoints. Reported results are averaged over 5 seeds, for which this is the expected level of agreement.
- **DataLoader workers.** `get_loaders(..., num_workers=N)` is configurable and does not affect results for the deterministic `without`/`full` augmentation modes.

## License

MIT — see [LICENSE](LICENSE). Datasets are downloaded via `torchvision` and are subject to their respective licenses (MNIST / FashionMNIST: [Creative Commons Attribution-Share Alike 3.0](https://creativecommons.org/licenses/by-sa/3.0/); CIFAR-10: [MIT](https://www.cs.toronto.edu/~kriz/cifar.html)).
