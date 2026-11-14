#!/bin/bash
#SBATCH -J rebuttal_histo
#SBATCH --gpus-per-node=1
#SBATCH -t 3:00:00
#SBATCH --array=0-69
#SBATCH --output=logs/rebuttal_histo_%A_%a.out
#SBATCH --error=logs/rebuttal_histo_%A_%a.err

# ----------------------------------------------------------------------------
# NCT-CRC-HE histology (zenodo.org/records/1214456): N0 = 1000 patches of NCT-CRC-HE-100K,
# tested on CRC-VAL-HE-7K, 9 classes, 224x224 RGB, full C4 augmentation.
# SGD (lr 1e-2, Nesterov, no weight decay), 500 epochs.
# 14 (method, epoch) configs x 5 seeds = 70 tasks, seed innermost;
#   epoch: trigger epoch of avg / proj, stage-1 epochs of gcnn (rest = stage 2).
# Submit from the repository root, after scripts/prepare_nct_crc_he.py (run once):
#   mkdir -p logs && sbatch run_rebuttal_histo_enforce.sh
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
    "avg 0"
    "proj 0"
    "gcnn 0"
    "avg 20"
    "proj 20"
    "gcnn 20"
    "baseline -"
    "proj_rescale 0"
    "avg_rescale 0"
    "proj_rescale 20"
    "avg_rescale 20"
    "avg 100"
    "proj_rescale 100"
    "gcnn 100"
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

# NCT-CRC-HE: five conv + pool blocks, 224 -> 7
TRAIN_SIZE=1000
IMAGE_SIZE=224
CONV_CHANNELS=(32 64 128 256 256)
VAL_INTERVAL=10   # validation is expensive at 224 px

# ----------------------------------------------------------------------------
# Index decomposition
# ----------------------------------------------------------------------------
if (( SLURM_ARRAY_TASK_ID >= NUM_CONFIGS * NUM_SEEDS )); then
    echo "Array task $SLURM_ARRAY_TASK_ID out of range (grid has $(( NUM_CONFIGS * NUM_SEEDS )) tasks)"
    exit 1
fi
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))]}
read -r METHOD SYM_EPOCH <<< "${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))]}"

for SPLIT in NCT-CRC-HE-100K CRC-VAL-HE-7K; do
    ARRAY="$REPO_DIR/data/NCT_CRC_HE/processed/${SPLIT}_${IMAGE_SIZE}px_images.npy"
    if [[ ! -f "$ARRAY" ]]; then
        echo "Missing $ARRAY -- run scripts/prepare_nct_crc_he.py first."
        exit 1
    fi
done

case "$METHOD" in
    baseline)
        METHOD_ARGS=(--method baseline --epochs "$EPOCHS") ;;
    avg)
        METHOD_ARGS=(--method avg --avg_method geometric --trigger_epoch "$SYM_EPOCH" --epochs "$EPOCHS") ;;
    proj)
        METHOD_ARGS=(--method proj --trigger_epoch "$SYM_EPOCH" --epochs "$EPOCHS") ;;
    avg_rescale)
        METHOD_ARGS=(--method avg --avg_method geometric --trigger_epoch "$SYM_EPOCH" --sym_rescale
                     --epochs "$EPOCHS") ;;
    proj_rescale)
        METHOD_ARGS=(--method proj --trigger_epoch "$SYM_EPOCH" --sym_rescale --epochs "$EPOCHS") ;;
    gcnn)
        METHOD_ARGS=(--method gcnn --arrangement gcnn --stage1_epochs "$SYM_EPOCH"
                     --stage2_epochs $(( EPOCHS - SYM_EPOCH ))) ;;
    *)
        echo "Unknown method: $METHOD"; exit 1 ;;
esac

if [[ "$METHOD" == baseline ]]; then
    RUN_NAME="NCT_CRC_HE_baseline_n${TRAIN_SIZE}"
else
    RUN_NAME="NCT_CRC_HE_${METHOD}_e${SYM_EPOCH}_n${TRAIN_SIZE}"
fi

echo "method=${METHOD} epoch=${SYM_EPOCH} seed=${SEED} train_size=${TRAIN_SIZE} optimizer=sgd lr=${LR}"
echo "run_name=${RUN_NAME} project=${PROJECT}"
echo "repo=${REPO_DIR}"
echo ""

apptainer exec --nv \
    -B "$REPO_DIR:/app" \
    -B "$REPO_DIR/wandb:/wandb" \
    "$CONTAINER" \
    env PYTHONPATH=/app WANDB_MODE=online WANDB_DIR=/wandb python "/app/$ENTRY" \
    "${METHOD_ARGS[@]}" \
    --dataset NCT_CRC_HE \
    --train_size "$TRAIN_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --conv_channels "${CONV_CHANNELS[@]}" \
    --val_interval "$VAL_INTERVAL" \
    --per_sample_backward \
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
