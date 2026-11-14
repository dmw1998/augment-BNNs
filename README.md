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
├── p4cnn.py              # Bayesian C4 G-CNN (Cohen & Welling), equivariant by construction
├── orbit.py              # orbit-averaged likelihood BNN (Nabarro et al., 2022), invariant by construction
├── data.py               # C4 augmentation datasets + get_loaders
├── histology.py          # NCT-CRC-HE histology dataset (download + preprocessing)
├── metrics.py            # equivariance defect, orbit consistency, symmetric KL (MC + non-MC)
├── symmetrize.py         # avg / proj / gcnn / enforce operators, optimiser + re-init helpers
└── training.py           # shared ELBO training + MC evaluation loops
scripts/
├── train_symmetrization.py  # main trainer: --method {avg, proj, gcnn, baseline, enforce, p4cnn, orbit}
├── prepare_nct_crc_he.py    # one-off NCT-CRC-HE download + preprocessing
├── train_theorem3.py        # random prior + invariant prior
├── train_theorem4.py        # T-sweep + K-fold MC defect (sample complexity)
├── train_multifamily.py     # variational-family comparison (closedness appendix)
└── summarize_logs.py        # SLURM logs -> markdown table (mean ± std over seeds)
tests/
└── test_smoke.py            # CPU smoke tests (no dataset / GPU required)
run_*.sh                     # SLURM array jobs (see "SLURM scripts" below)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `ebnn` package (editable) and its dependencies
(`torch`, `torchvision`, `numpy`, `wandb`). A CUDA-enabled PyTorch build is recommended for training; the code falls back to CPU automatically.

Datasets (MNIST / FashionMNIST / CIFAR-10) are downloaded on first use via `torchvision` into `./data`. The C₄-augmented and equivariance-evaluation sets are precomputed and cached under `./data/precomputed_aug`. The NCT-CRC-HE histology set is fetched from Zenodo by `scripts/prepare_nct_crc_he.py` (see [below](#histology-nct-crc-he)).

## Reproducing the experiments

All training runs log to [Weights & Biases](https://wandb.ai). Pass `--no_wandb` to disable logging, or set the project with `--project`. The main results are averaged over 5 seeds; the example commands below use a single seed.

### Inducing equivariance (main results)

The strongest configuration in the paper is orbit expansion (`gcnn`) trained with SGD on FashionMNIST (N₀ = 5000, full C₄ augmentation). **This is the default configuration**, so the bare command reproduces it (gcnn + SGD, `conv_channels [32, 64]`, i.e. a stage-1 model with `[8, 16]` channels expanded to `[32, 64]`, 100 + 400 stage epochs, `lr 1e-2`, `weight_decay 0`, Nesterov, `seed 13`):

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

**Validation.** Every validation step makes one MC pass over the C₄ orbit of every test image (`ebnn.metrics.evaluate_c4_orbit`) and logs, from the same predictive draws, the accuracy/NLL on the C₄-augmented test set (`val/accuracy_c4`), on the original test images (`val/accuracy_orig`), OSP and the symmetric KL. `--val_source` picks which accuracy is `val/accuracy` and selects the best checkpoint: `equiv_all` (default, C₄-augmented test set) or `test` (original images).

`--conv_channels` is the number of feature-map channels per layer for **every** method, so all methods are compared at the same width. For `gcnn` it is the stage-2 width: the stage-1 small model has `conv_channels // group_size` channels and the orbit expansion multiplies them by `group_size` (`32 64` → stage 1 `8 16` → stage 2 `32 64`). Each entry must be divisible by `group_size`.

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

### Closedness of the exponential family (appendix)

The mean-field Gaussian family is closed under the C₄ push-forward, so the
equivariance results apply; Laplace / Log-Normal are not closed (for Log-Normal,
the fixed random sign pattern needed to represent negative weights is not
rotation-invariant). This experiment trains the same architecture with each
variational family and compares their equivariance metrics. Run all three
families with the same SGD setup as the main results (same architecture, N₀,
augmentation, epochs, seeds), varying only `--family`:

```bash
# family in {gaussian, laplace, lognormal}; vary --seed for the 5-seed average
python scripts/train_multifamily.py --family gaussian \
    --dataset FashionMNIST --train_size 5000 --conv_channels 32 64 --epochs 500 \
    --optimizer sgd --lr 1e-2 --weight_decay 0 --nesterov --seed 13
```

Supports KL annealing (`--kl_annealing_epochs`, `--kl_annealing_epochs_fc`),
free bits (`--free_bits`), separate conv/fc KL weights, and `--no_augmentation`
for the no-augmentation baseline. (The default optimiser is AdamW; pass
`--optimizer sgd` as above to match the SGD results in the main text.)

### Enforcing equivariance throughout training (`enforce` baseline)

Instead of symmetrising once, `--method enforce` keeps the posterior on the
C₄-equivariant subspace for the whole run. The operator chosen by `--enforce_op`
(`proj`, or `avg` with `--avg_method`) is applied right after initialisation,
after which every layer's posterior mean and std are rescaled back to their
pre-symmetrisation Frobenius norms (projection / averaging shrinks i.i.d. random
weights by ≈2× / ≈√3×). The same operator is then re-applied after every
optimizer step, i.e. projected SGD; it is vectorised, so the per-step cost is
negligible. `--enforce_rescale {init,always,none}` selects when the norm is
restored (default `init`).

```bash
python scripts/train_symmetrization.py --method enforce --enforce_op proj
python scripts/train_symmetrization.py --method enforce --enforce_op avg --avg_method geometric
```

For `proj`, the enforced projection also averages the 1×1 classifier's posterior
*scale* over each C₄ group, so the posterior is exactly invariant in distribution
(the one-shot `proj` averages only the classifier mean).

### Equivariant-by-construction BNN (`p4cnn` baseline)

`--method p4cnn` trains a Bayesian C₄ group-equivariant CNN (P4CNN; Cohen &
Welling, 2016) from scratch: a lifting convolution, P4 group convolutions and a
rotation-averaged (invariant) 1×1 classifier. Unlike the `gcnn` method, whose
weights are untied after the orbit expansion, the mean-field posterior lives on the
free filters and every layer builds its full weight by rotating / cyclically
shifting them, so each posterior sample is an exactly C₄-equivariant network.
With `--conv_channels` C it has C/4 fields of 4 rotations each, i.e. the same
feature-map widths as every other method; for proj / enforce-proj the posterior
mean lies in exactly this function class. Each free filter is shared by the 4
rotations, so a group layer has ¼ of the parameters of the corresponding `gcnn` layer.

```bash
python scripts/train_symmetrization.py --method p4cnn          # FashionMNIST, conv_channels [32, 64]
```

### Orbit-averaged likelihood (`orbit` baseline)

`--method orbit` implements the orbit-averaged likelihood of Nabarro et al. (UAI 2022,
"Data augmentation in Bayesian neural networks and the cold posterior effect") on the
same `BayesianFullyConvCNN` as `avg` / `proj` / `baseline`: every input is evaluated at
its 4 rotations with **one** shared posterior sample, and the predictions are averaged
in probability space (`--orbit_lik prob`, p(y|x,θ) = ¼ Σ_g softmax f_θ(gx)) or in logit
space (`--orbit_lik logit`). Every posterior sample is therefore exactly C₄-invariant,
at 4× the forward cost of the base network in training and evaluation. Initialisation,
optimiser, epochs, full C₄ augmentation (a no-op for this model) and the KL weight
(1 / len(train set) = 1 / (4 N₀); pass `--beta` for another value, e.g. Nabarro et al.'s
1 / N₀) are those of every other method, so only the likelihood differs.

```bash
python scripts/train_symmetrization.py --method orbit --orbit_lik prob    # or logit
```

### Histology: NCT-CRC-HE

[NCT-CRC-HE](https://zenodo.org/records/1214456) (Kather et al., 2018): 224×224
H&E patches of colorectal cancer and normal tissue in 9 classes; training uses
NCT-CRC-HE-100K (100,000 patches) and testing CRC-VAL-HE-7K (7,180 patches from
50 other patients). Tissue has no canonical orientation, so C₄ rotations are an
exact symmetry of the labels, and every method (`baseline`, `avg`, `proj`, `gcnn`,
`enforce`, `p4cnn`, `orbit`) applies unchanged with `--dataset NCT_CRC_HE`. Prepare the data once
(12.5 GB download, decoded into uint8 arrays that later runs memory-map):

```bash
python scripts/prepare_nct_crc_he.py                  # native 224 px; --image_size N to resize
python scripts/train_symmetrization.py --dataset NCT_CRC_HE --method gcnn --per_sample_backward
```

Defaults for this dataset are `--image_size 224` and `conv_channels [32, 64, 128, 256, 256]`
(five conv + pool blocks, 224 → 7). Keep the input size divisible by 2^(#blocks) so that
max-pooling stays exactly C₄-equivariant. At 224 px pass `--per_sample_backward`, which
backpropagates each MC sample separately: the gradient is identical and activation
memory drops by ≈`train_samples`×. The orbit sets for this dataset are built in
memory rather than cached under `./data/precomputed_aug`.

### SLURM scripts

Array jobs (5 seeds each, SGD, 500 epochs), submitted from the repository root. The
scripts contain no cluster-specific settings; provide them through the environment
(`sbatch` reads `SBATCH_*` variables, and the jobs inherit `CONTAINER`):

```bash
export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<gpu-partition>   # optional: SBATCH_GPUS_PER_NODE=<type>:1
export CONTAINER=/path/to/image.sif                                 # apptainer image with the requirements
mkdir -p logs && sbatch <script>
``` `python scripts/summarize_logs.py "logs/<prefix>_*.out"` turns the logs
into a table.

| Script | Experiments |
|---|---|
| `run_gcnn_grid.sh` | default `gcnn` (100 + 400 epochs), FashionMNIST |
| `run_enforce_baseline.sh` | FashionMNIST: `baseline`, `avg` / `proj` (± `--sym_rescale`) and `gcnn` at epoch 0 / 20 / 100, `enforce` |
| `run_rebuttal_histo_enforce.sh` | NCT-CRC-HE: `baseline`, `avg` / `proj` (± `--sym_rescale`) and `gcnn` at epoch 0 / 20 / 100 |
| `run_p4cnn_baseline.sh` | `p4cnn` on FashionMNIST and NCT-CRC-HE |
| `run_orbit_baseline.sh` | `orbit` (prob / logit) on FashionMNIST and NCT-CRC-HE, plus a compute-matched FashionMNIST variant |
| `run_closedness.sh` | variational-family comparison (closedness appendix) |

### Mapping to the paper's tables and figures

All symmetrization runs use the default setup (FashionMNIST, N₀ = 5000, full C₄
augmentation, `conv_channels [32, 64]`, 500 total epochs, SGD, 5 seeds; vary
`--seed`). In the symmetrization tables the **Epoch** column is the trigger epoch
(`--trigger_epoch`, where `0` = right after init) for `avg`/`proj`, and the number
of Stage-1 epochs (`--stage1_epochs`, with `--stage2_epochs` = 500 − that) for
`gcnn`.

**Figure 2 — theorem validation** (sweep `--train_size` over {500, 5000, 20000, 50000}):

```bash
python scripts/train_theorem3.py --train_size 5000                  # (a) invariant prior
python scripts/train_theorem3.py --train_size 5000 --random_prior   # (a) random prior
python scripts/train_theorem4.py --train_size 5000 --eval_samples 1024 --K_mc_runs 10 --skip_single_sweep  # (b,c)
```

**Table 1 — main symmetrization comparison** (SGD; Epoch ∈ {0, 20, 100}):

```bash
python scripts/train_symmetrization.py --method baseline
python scripts/train_symmetrization.py --method avg  --avg_method geometric --trigger_epoch 20
python scripts/train_symmetrization.py --method proj --trigger_epoch 20
python scripts/train_symmetrization.py --method gcnn --stage1_epochs 100 --stage2_epochs 400  # = bare default
```

**Table 4 — trigger-timing × optimizer** (Table 1 runs plus `--optimizer adamw`; Epoch ∈ {0, 20, 50, 100, 150}):

```bash
python scripts/train_symmetrization.py --method proj --optimizer adamw --trigger_epoch 50
python scripts/train_symmetrization.py --method gcnn --optimizer adamw --stage1_epochs 50 --stage2_epochs 450
```

**Table 5 — orbit-expansion filter-arrangement ablation** (SGD; `--arrangement`; Epoch ∈ {0, 20, 100}):

```bash
python scripts/train_symmetrization.py --method gcnn --arrangement gcnn         --stage1_epochs 20 --stage2_epochs 480
python scripts/train_symmetrization.py --method gcnn --arrangement row_rolled   --stage1_epochs 20 --stage2_epochs 480
python scripts/train_symmetrization.py --method gcnn --arrangement row_constant --stage1_epochs 20 --stage2_epochs 480
```

## Tests

```bash
python tests/test_smoke.py        # or: python -m pytest tests/test_smoke.py
```

The smoke tests run on CPU without any dataset (model construction, invariance of the symmetrisation operators and of the P4CNN / orbit models, validation metrics, NCT-CRC-HE preprocessing on a synthetic zip).

## Notes on reproducibility

- **Posterior initialisation regimes.** Two initialisations are used and are preserved exactly via explicit layer arguments. The Theorem-3/4 baselines use Kaiming-initialised posterior means with `softplus(rho_init)` scale; the fully-/group-convolutional models use zero means with `softplus(0.693) ≈ 1` scale (and the unified trainer additionally re-initialises means/scales at startup via `initialize_posterior_mean` / `initialize_gaussian_posterior_scale`).
- **Functional, not bit-exact.** This is a cleaned/restructured version of the original research code. It is numerically equivalent to the original (same architectures, KL terms, ELBO, and training math), so re-running reproduces results up to the usual seed and hardware (CUDA) non-determinism. It is **not** guaranteed to be bit-identical to the original checkpoints. Reported results are averaged over 5 seeds, for which this is the expected level of agreement.
- **DataLoader workers.** `get_loaders(..., num_workers=N)` is configurable and does not affect results for the deterministic `without`/`full` augmentation modes.

## License

MIT — see [LICENSE](LICENSE). Datasets are downloaded via `torchvision` and are subject to their respective licenses (MNIST / FashionMNIST: [Creative Commons Attribution-Share Alike 3.0](https://creativecommons.org/licenses/by-sa/3.0/); CIFAR-10: [MIT](https://www.cs.toronto.edu/~kriz/cifar.html); NCT-CRC-HE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), downloaded from Zenodo).
