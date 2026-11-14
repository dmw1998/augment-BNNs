#!/bin/bash
#SBATCH -J rebuttal_p4cnn
#SBATCH --gpus-per-node=1
#SBATCH -t 4:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/rebuttal_p4cnn_%A_%a.out
#SBATCH --error=logs/rebuttal_p4cnn_%A_%a.err

# ----------------------------------------------------------------------------
# Bayesian C4 G-CNN (Cohen & Welling 2016; ebnn/p4cnn.py), equivariant by construction,
# trained from scratch for 500 epochs with the same width, augmentation, optimiser and
# evaluation as the other methods. Tasks 0-4: FashionMNIST, 5-9: NCT-CRC-HE (5 seeds each).
# Submit from the repository root:  mkdir -p logs && sbatch run_p4cnn_baseline.sh
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
# Grid: one "dataset seed" entry per task
# ----------------------------------------------------------------------------
TASKS=(
    "FashionMNIST 2026"
    "FashionMNIST 529"
    "FashionMNIST 1507"
    "FashionMNIST 13"
    "FashionMNIST 25"
    "NCT_CRC_HE 2026"
    "NCT_CRC_HE 529"
    "NCT_CRC_HE 1507"
    "NCT_CRC_HE 13"
    "NCT_CRC_HE 25"
)

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

# NCT-CRC-HE: five P4 conv + pool blocks, 224 -> 7
HISTO_TRAIN_SIZE=1000
HISTO_IMAGE_SIZE=224
HISTO_CONV_CHANNELS=(32 64 128 256 256)
HISTO_VAL_INTERVAL=10   # validation is expensive at 224 px

# ----------------------------------------------------------------------------
# Index decomposition
# ----------------------------------------------------------------------------
if (( SLURM_ARRAY_TASK_ID >= ${#TASKS[@]} )); then
    echo "Array task $SLURM_ARRAY_TASK_ID out of range (grid has ${#TASKS[@]} tasks)"
    exit 1
fi
read -r DATASET SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

case "$DATASET" in
    NCT_CRC_HE)
        DATA_ARGS=(--train_size "$HISTO_TRAIN_SIZE" --image_size "$HISTO_IMAGE_SIZE"
                   --conv_channels "${HISTO_CONV_CHANNELS[@]}" --val_interval "$HISTO_VAL_INTERVAL"
                   --per_sample_backward)
        for SPLIT in NCT-CRC-HE-100K CRC-VAL-HE-7K; do
            ARRAY="$REPO_DIR/data/NCT_CRC_HE/processed/${SPLIT}_${HISTO_IMAGE_SIZE}px_images.npy"
            if [[ ! -f "$ARRAY" ]]; then
                echo "Missing $ARRAY -- run scripts/prepare_nct_crc_he.py first."
                exit 1
            fi
        done
        RUN_NAME="NCT_CRC_HE_p4cnn_n${HISTO_TRAIN_SIZE}"
        ;;
    FashionMNIST)
        DATA_ARGS=(--train_size "$FMNIST_TRAIN_SIZE" --conv_channels "${FMNIST_CONV_CHANNELS[@]}"
                   --val_interval "$FMNIST_VAL_INTERVAL")
        RUN_NAME="FashionMNIST_p4cnn"
        ;;
    *)
        echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

echo "dataset=${DATASET} method=p4cnn seed=${SEED} optimizer=sgd lr=${LR}"
echo "run_name=${RUN_NAME} project=${PROJECT}"
echo "repo=${REPO_DIR}"
echo ""

apptainer exec --nv \
    -B "$REPO_DIR:/app" \
    -B "$REPO_DIR/wandb:/wandb" \
    "$CONTAINER" \
    env PYTHONPATH=/app WANDB_MODE=online WANDB_DIR=/wandb python "/app/$ENTRY" \
    --method p4cnn \
    --epochs "$EPOCHS" \
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
