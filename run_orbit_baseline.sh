#!/bin/bash
#SBATCH -J rebuttal_orbit
#SBATCH --gpus-per-node=1
#SBATCH -t 8:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/rebuttal_orbit_%A_%a.out
#SBATCH --error=logs/rebuttal_orbit_%A_%a.err

# ----------------------------------------------------------------------------
# Orbit-averaged likelihood (Nabarro et al., UAI 2022; ebnn/orbit.py): the network of
# avg / proj with one posterior sample applied to all 4 rotations of every input,
# averaged in probability (prob) or logit (logit) space. 6 configs x 5 seeds, seed innermost:
#   0-9   FashionMNIST prob / logit      10-19  NCT-CRC-HE prob / logit
#   20-29 FashionMNIST prob / logit, compute-matched (--train_samples 2 --eval_samples 4,
#         i.e. no more forward passes than the other methods)
# Submit from the repository root:  mkdir -p logs && sbatch run_orbit_baseline.sh
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
# Grid: (dataset, likelihood) configs x 5 seeds (seed innermost)
# ----------------------------------------------------------------------------
SEEDS=(2026 529 1507 13 25)
NUM_SEEDS=${#SEEDS[@]}

CONFIGS=(
    "FashionMNIST prob"
    "FashionMNIST logit"
    "NCT_CRC_HE prob"
    "NCT_CRC_HE logit"
    "FashionMNIST prob matched"
    "FashionMNIST logit matched"
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

# NCT-CRC-HE: five conv + pool blocks, 224 -> 7
HISTO_TRAIN_SIZE=1000
HISTO_IMAGE_SIZE=224
HISTO_CONV_CHANNELS=(32 64 128 256 256)
HISTO_VAL_INTERVAL=10   # validation is expensive at 224 px

# ----------------------------------------------------------------------------
# Index decomposition
# ----------------------------------------------------------------------------
if (( SLURM_ARRAY_TASK_ID >= NUM_CONFIGS * NUM_SEEDS )); then
    echo "Array task $SLURM_ARRAY_TASK_ID out of range (grid has $(( NUM_CONFIGS * NUM_SEEDS )) tasks)"
    exit 1
fi
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))]}
read -r DATASET LIK BUDGET <<< "${CONFIGS[$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))]}"

# Compute-matched budget (see header)
TAG="${LIK}"
if [[ "$BUDGET" == matched ]]; then
    TRAIN_SAMPLES=2
    EVAL_SAMPLES=4
    TAG="${LIK}_matched"
fi

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
        RUN_NAME="NCT_CRC_HE_orbit_${TAG}_n${HISTO_TRAIN_SIZE}"
        ;;
    FashionMNIST)
        DATA_ARGS=(--train_size "$FMNIST_TRAIN_SIZE" --conv_channels "${FMNIST_CONV_CHANNELS[@]}"
                   --val_interval "$FMNIST_VAL_INTERVAL")
        RUN_NAME="FashionMNIST_orbit_${TAG}"
        ;;
    *)
        echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

echo "dataset=${DATASET} method=orbit_${TAG} seed=${SEED} optimizer=sgd lr=${LR}"
echo "train_samples=${TRAIN_SAMPLES} eval_samples=${EVAL_SAMPLES}"
echo "run_name=${RUN_NAME} project=${PROJECT}"
echo "repo=${REPO_DIR}"
echo ""

apptainer exec --nv \
    -B "$REPO_DIR:/app" \
    -B "$REPO_DIR/wandb:/wandb" \
    "$CONTAINER" \
    env PYTHONPATH=/app WANDB_MODE=online WANDB_DIR=/wandb python "/app/$ENTRY" \
    --method orbit \
    --orbit_lik "$LIK" \
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
