#!/bin/bash
#SBATCH -J rebuttal_fmnist
#SBATCH --gpus-per-node=1
#SBATCH -t 3:00:00
#SBATCH --array=0-89
#SBATCH --output=logs/rebuttal_%A_%a.out
#SBATCH --error=logs/rebuttal_%A_%a.err

# ----------------------------------------------------------------------------
# FashionMNIST (N0 = 5000), SGD (lr 1e-2, Nesterov, no weight decay), 500 epochs.
# 18 configs x 5 seeds = 90 tasks, seed innermost. Config: dataset method rescale epoch
#   epoch:   trigger epoch of avg / proj, stage-1 epochs of gcnn (rest = stage 2)
#   rescale: --enforce_rescale of enforce_*; *_rescale = one-shot op + --sym_rescale
# Submit from the repository root:  mkdir -p logs && sbatch run_enforce_baseline.sh
# ----------------------------------------------------------------------------

echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# ----------------------------------------------------------------------------
# Repo + container (the repo root goes on PYTHONPATH so that `import ebnn` works)
# ----------------------------------------------------------------------------
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR" || exit 1
mkdir -p "$REPO_DIR/wandb"

CONTAINER="${CONTAINER:?set CONTAINER to the apptainer image (see README)}"
ENTRY=scripts/train_symmetrization.py

# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
SEEDS=(2026 529 1507 13 25)
NUM_SEEDS=${#SEEDS[@]}

CONFIGS=(
    "FashionMNIST enforce_proj none 0"
    "FashionMNIST enforce_avg none 0"
    "FashionMNIST enforce_proj init 0"
    "FashionMNIST enforce_avg init 0"
    "FashionMNIST enforce_proj always 0"
    "FashionMNIST enforce_avg always 0"
    "FashionMNIST baseline none 0"
    "FashionMNIST avg none 0"
    "FashionMNIST proj none 0"
    "FashionMNIST gcnn none 0"
    "FashionMNIST avg_rescale free 0"
    "FashionMNIST proj_rescale free 0"
    "FashionMNIST avg none 20"
    "FashionMNIST avg none 100"
    "FashionMNIST proj_rescale free 20"
    "FashionMNIST proj_rescale free 100"
    "FashionMNIST gcnn none 20"
    "FashionMNIST gcnn none 100"
)
NUM_CONFIGS=${#CONFIGS[@]}

# Fixed config
EPOCHS=500
TRAIN_SAMPLES=10
EVAL_SAMPLES=16
AUGMENTATION=full
LR=1e-2                 # SGD learning rate (rho group gets LR * rho_lr_mult)
RHO_LR_MULT=10
PROJECT=rebuttal_histological_and_equiv_enforce_BNN_2

# FashionMNIST
FMNIST_TRAIN_SIZE=5000
FMNIST_CONV_CHANNELS=(32 64)
FMNIST_VAL_INTERVAL=5

# ----------------------------------------------------------------------------
# Index decomposition
# ----------------------------------------------------------------------------
if (( SLURM_ARRAY_TASK_ID >= NUM_CONFIGS * NUM_SEEDS )); then
    echo "Array task $SLURM_ARRAY_TASK_ID out of range (grid has $(( NUM_CONFIGS * NUM_SEEDS )) tasks)"
    exit 1
fi
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))]}
read -r DATASET METHOD ENFORCE_RESCALE TRIGGER_EPOCH <<< "${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))]}"

case "$DATASET" in
    FashionMNIST)
        DATA_ARGS=(--train_size "$FMNIST_TRAIN_SIZE" --conv_channels "${FMNIST_CONV_CHANNELS[@]}"
                   --val_interval "$FMNIST_VAL_INTERVAL")
        ;;
    *)
        echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

case "$METHOD" in
    baseline)
        METHOD_ARGS=(--method baseline --epochs "$EPOCHS") ;;
    avg)
        METHOD_ARGS=(--method avg --avg_method geometric --trigger_epoch "$TRIGGER_EPOCH" --epochs "$EPOCHS") ;;
    proj)
        METHOD_ARGS=(--method proj --trigger_epoch "$TRIGGER_EPOCH" --epochs "$EPOCHS") ;;
    avg_rescale)
        METHOD_ARGS=(--method avg --avg_method geometric --trigger_epoch "$TRIGGER_EPOCH" --sym_rescale
                     --epochs "$EPOCHS") ;;
    proj_rescale)
        METHOD_ARGS=(--method proj --trigger_epoch "$TRIGGER_EPOCH" --sym_rescale --epochs "$EPOCHS") ;;
    gcnn)
        METHOD_ARGS=(--method gcnn --arrangement gcnn --stage1_epochs "$TRIGGER_EPOCH"
                     --stage2_epochs $(( EPOCHS - TRIGGER_EPOCH ))) ;;
    enforce_proj)
        METHOD_ARGS=(--method enforce --enforce_op proj --enforce_rescale "$ENFORCE_RESCALE"
                     --epochs "$EPOCHS") ;;
    enforce_avg)
        METHOD_ARGS=(--method enforce --enforce_op avg --avg_method geometric
                     --enforce_rescale "$ENFORCE_RESCALE" --epochs "$EPOCHS") ;;
    *)
        echo "Unknown method: $METHOD"; exit 1 ;;
esac

RUN_NAME="${DATASET}_${METHOD}_${TRIGGER_EPOCH}_${ENFORCE_RESCALE}_shuf"

echo "dataset=${DATASET} method=${METHOD} seed=${SEED} optimizer=sgd lr=${LR}"
echo "run_name=${RUN_NAME} project=${PROJECT}"
echo "repo=${REPO_DIR}"
echo ""

apptainer exec --nv \
    -B "$REPO_DIR:/app" \
    -B "$REPO_DIR/wandb:/wandb" \
    "$CONTAINER" \
    env PYTHONPATH=/app WANDB_MODE=online WANDB_DIR=/wandb python "/app/$ENTRY" \
    "${METHOD_ARGS[@]}" \
    --dataset "$DATASET" \
    "${DATA_ARGS[@]}" \
    --optimizer sgd \
    --batch_size 128 \
    --seed "$SEED" \
    --posterior_init kaiming \
    --train_samples "$TRAIN_SAMPLES" \
    --eval_samples "$EVAL_SAMPLES" \
    --augmentation "$AUGMENTATION" \
    --lr "$LR" \
    --weight_decay 0 \
    --rho_weight_decay 0 \
    --rho_lr_mult "$RHO_LR_MULT" \
    --nesterov \
    --project "$PROJECT" \
    --run_name "$RUN_NAME" \
    --save_model \
    --save_path "/app/trained_models/${PROJECT}/${RUN_NAME}_s${SEED}.pt"

echo ""
echo "End time: $(date)"