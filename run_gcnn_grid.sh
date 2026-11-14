#!/bin/bash
#SBATCH -J gcnn_sgd
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 03:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/gcnn_sgd_%A_%a.out
#SBATCH --error=logs/gcnn_sgd_%A_%a.err

# ----------------------------------------------------------------------------
# Default configuration: gcnn (100 stage-1 + 400 stage-2 epochs) + SGD on FashionMNIST
# (N0 = 5000), one task per seed.
# Submit from the repository root:  mkdir -p logs && sbatch run_gcnn_grid.sh
# ----------------------------------------------------------------------------

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# ----------------------------------------------------------------------------
# Repo + container
# ----------------------------------------------------------------------------
# The repo root goes on PYTHONPATH so that `import ebnn` works.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR" || exit 1
mkdir -p "$REPO_DIR/wandb"

CONTAINER="${CONTAINER:?set CONTAINER to the apptainer image (see README)}"
ENTRY=scripts/train_symmetrization.py

# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
SEEDS=(2026 529 1507 13 25)

# Fixed config
EPOCHS=500
DATASET=FashionMNIST
TRAIN_SIZE=5000
CONV_CHANNELS="32 64"
TRAIN_SAMPLES=10
EVAL_SAMPLES=16
AUGMENTATION=full
PROJECT=rebuttal_histological_and_equiv_enforce_BNN_2

# SGD learning rate (rho group gets LR * rho_lr_mult)
LR_SGD=1e-2
RHO_LR_MULT=10

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="gcnn_sgd_s${SEED}"

echo "seed=${SEED} optimizer=sgd method=gcnn lr=${LR_SGD}"
echo "run_name=${RUN_NAME}"
echo "repo=${REPO_DIR}"
echo ""

apptainer exec --nv \
    -B "$REPO_DIR:/app" \
    -B "$REPO_DIR/wandb:/wandb" \
    "$CONTAINER" \
    env PYTHONPATH=/app WANDB_MODE=online WANDB_DIR=/wandb python "/app/$ENTRY" \
    --method gcnn \
    --optimizer sgd \
    --dataset ${DATASET} \
    --train_size ${TRAIN_SIZE} \
    --batch_size 128 \
    --seed ${SEED} \
    --posterior_init kaiming \
    --train_samples ${TRAIN_SAMPLES} \
    --eval_samples ${EVAL_SAMPLES} \
    --augmentation ${AUGMENTATION} \
    --conv_channels ${CONV_CHANNELS} \
    --epochs ${EPOCHS} \
    --lr ${LR_SGD} \
    --weight_decay 0 \
    --rho_weight_decay 0 \
    --rho_lr_mult ${RHO_LR_MULT} \
    --project ${PROJECT} \
    --run_name ${RUN_NAME} \
    --stage1_epochs 100 \
    --stage2_epochs 400 \
    --nesterov \
    --save_model \
    --save_path "/app/trained_models/${PROJECT}/${RUN_NAME}.pt"

echo ""
echo "End time: $(date)"
