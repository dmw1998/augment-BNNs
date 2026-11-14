#!/bin/bash
#SBATCH -J closedness_%a
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 04:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/closedness_%A_%a.out
#SBATCH --error=logs/closedness_%A_%a.err

# ----------------------------------------------------------------------------
# Closedness-of-exponential-family appendix.
# Trains the SAME architecture with three variational families and compares
# their equivariance:
#   gaussian, laplace  -> closed under the C4 push-forward (positive examples)
#   lognormal          -> NOT closed (fixed random signs are not rotation-
#                         invariant); the counterexample.
# All runs use SGD, to match the main-text results.
#
# BEFORE SUBMITTING:
#   mkdir -p logs
#   submit from the repository ROOT (the folder containing ebnn/ and scripts/).
# ----------------------------------------------------------------------------

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# ---- grid: 3 families x 5 seeds = 15 jobs ----
SEEDS=(2026 529 1507 13 25)
NUM_SEEDS=${#SEEDS[@]}

FAMILY=(gaussian laplace lognormal)
NUM_FAMILY=${#FAMILY[@]}

# ---- fixed config (aligned with the main-text SGD runs) ----
EPOCHS=500
DATASET=FashionMNIST
TRAIN_SIZE=5000
CONV_CHANNELS="32 64"
TRAIN_SAMPLES=10
EVAL_SAMPLES=16
LR=1e-2                 # SGD learning rate (rho group gets LR * rho_lr_mult)
RHO_LR_MULT=10
PROJECT=closedness_appendix
CONTAINER="${CONTAINER:?set CONTAINER to the apptainer image (see README)}"

# ---- index decomposition (seed innermost) ----
SEED_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
FAMILY_INDEX=$(( (SLURM_ARRAY_TASK_ID / NUM_SEEDS) % NUM_FAMILY ))

SEED=${SEEDS[$SEED_INDEX]}
FAM=${FAMILY[$FAMILY_INDEX]}

RUN_NAME="multifamily_sgd_${FAM}_seed${SEED}"

echo "family=${FAM} seed=${SEED} optimizer=sgd lr=${LR}"
echo "run_name=${RUN_NAME}"
echo ""

# Make the 'ebnn' package importable inside the container. The repository uses a
# package layout, so the repo ROOT must be on PYTHONPATH (running the script from
# scripts/ only puts scripts/ on sys.path). $SLURM_SUBMIT_DIR is the directory
# you ran `sbatch` from, i.e. the repo root.
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT"; exit 1; }

# If your apptainer is old and does not support --env, replace the line below with:
#   APPTAINERENV_PYTHONPATH="$REPO_ROOT" apptainer exec "$CONTAINER" python scripts/train_multifamily.py ...
apptainer exec --env PYTHONPATH="$REPO_ROOT" "$CONTAINER" \
    python scripts/train_multifamily.py \
        --family "${FAM}" \
        --dataset "${DATASET}" \
        --train_size "${TRAIN_SIZE}" \
        --conv_channels ${CONV_CHANNELS} \
        --epochs "${EPOCHS}" \
        --optimizer sgd \
        --lr "${LR}" \
        --weight_decay 0 \
        --nesterov \
        --rho_lr_mult "${RHO_LR_MULT}" \
        --train_samples "${TRAIN_SAMPLES}" \
        --eval_samples "${EVAL_SAMPLES}" \
        --seed "${SEED}" \
        --project "${PROJECT}" \
        --run_name "${RUN_NAME}"

echo ""
echo "End time: $(date)"
