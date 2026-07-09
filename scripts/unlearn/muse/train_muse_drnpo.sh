#!/bin/bash

DATE=$(date "+%m%d")
TIME=$(date "+%H%M%S")

# Default multi-GPU list; can be overridden from command line
# e.g. CUDA_VISIBLE_DEVICES=4,5 bash scripts/unlearn/muse/train_muse_drnpo.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//,/,}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
export CUDA_VISIBLE_DEVICES
export WANDB__DISABLE_STATS=true

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_SLOTS=${#GPU_IDS[@]}
if [ "$NUM_SLOTS" -lt 1 ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES parse failed: $CUDA_VISIBLE_DEVICES"
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    for gpu in "${GPU_IDS[@]}"; do
        if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
            echo "ERROR: invalid GPU id '$gpu'"
            exit 1
        fi
        if [ "$gpu" -ge "$GPU_COUNT" ]; then
            echo "ERROR: GPU '$gpu' out of range. Available GPU count: ${GPU_COUNT}."
            exit 1
        fi
    done
fi

STAGGER_SEC=60

REPORTTO="wandb"
WANDB_PROJECT="BalDRO-muse-News"

MODEL="Llama-2-7b-hf"
TRAINER="DrNPO"
splits=(
    "News"
    # "Books"
)

# lr, batchsize, grad_acc, epochs
# lr_set=("1e-5" "2e-5" "3e-5" "4e-5" "5e-5")
# bz_set=("4 4" "4 8")
# epoch_set=(10)
# beta_dv_set=(0.5 1.0 2.0 5.0 10.0)

lr_set=("5e-5")
bz_set=("4 8")
epoch_set=(10)
beta_dv_set=(0.5 1.0 2.0 5.0 10.0)

run_one() {
    local gpu=$1 split=$2 lr=$3 bz=$4 epochs=$5 beta_dv_forget=$6 run_idx=$7
    local PRETRAINED_PATH="muse-bench/MUSE-${split}_target"
    local TOKENIZER_PRETRAINED="meta-llama/Llama-2-7b-chat-hf"
    local bsz grad_acc
    bsz=$(echo "$bz" | cut -d' ' -f1)
    grad_acc=$(echo "$bz" | cut -d' ' -f2)

    local SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_e${epochs}_betaDV${beta_dv_forget}_day${DATE}_time${TIME}"
    local TASK_NAME="unlearn_muse_${split}_${MODEL}_${TRAINER}_${SUFFIX}"
    local OUTPUT_DIR="./saves/unlearn/muse/${split}/${MODEL}/${TRAINER}/${SUFFIX}"
    local LOG_FILE="./saves/logs/${TRAINER}_${split}_${SUFFIX}_gpu${gpu}.log"

    mkdir -p "$(dirname "$LOG_FILE")"
    echo "=========================================="
    echo "RUN ${run_idx} | GPU ${gpu}: ${SUFFIX}"
    echo "=========================================="

    export WANDB_PROJECT=${WANDB_PROJECT}
    CUDA_VISIBLE_DEVICES=$gpu python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        trainer=${TRAINER} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${PRETRAINED_PATH} \
        model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PRETRAINED} \
        data_split=${split} \
        task_name=${TASK_NAME} \
        paths.output_dir="${OUTPUT_DIR}" \
        do_save=False \
        eval.muse.retain_logs_path=./saves/eval/muse_${MODEL}_${split}_retrain/MUSE_EVAL.json \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.report_to=${REPORTTO} \
        trainer.args.run_name=${TASK_NAME} \
        trainer.args.logging_steps=1 \
        trainer.args.learning_rate=$lr \
        trainer.args.per_device_train_batch_size=$bsz \
        trainer.args.gradient_accumulation_steps=$grad_acc \
        trainer.args.num_train_epochs=$epochs \
        trainer.args.eval_strategy=epoch \
        trainer.args.eval_on_start=False \
        trainer.method_args.gamma=1.0 \
        trainer.method_args.alpha=1.0 \
        trainer.method_args.retain_loss_type=NLL \
        trainer.method_args.beta=0.1 \
        trainer.method_args.retain_dro=False \
        trainer.method_args.beta_dv_forget=${beta_dv_forget} \
        2>&1 | tee "$LOG_FILE"
}

total=0
fail_count=0
for split in "${splits[@]}"; do
    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            for epochs in "${epoch_set[@]}"; do
                for beta_dv_forget in "${beta_dv_set[@]}"; do
                    gpu=${GPU_IDS[$((total % NUM_SLOTS))]}
                    run_one "$gpu" "$split" "$lr" "$bz" "$epochs" "$beta_dv_forget" "$((total + 1))" &
                    total=$((total + 1))

                    if (( total % NUM_SLOTS == 0 )); then
                        for pid in $(jobs -p); do
                            wait "$pid" || fail_count=$((fail_count + 1))
                        done
                    else
                        sleep $STAGGER_SEC
                    fi
                done
            done
        done
    done
done

for pid in $(jobs -p); do
    wait "$pid" || fail_count=$((fail_count + 1))
done

echo "=========================================="
echo "ALL DONE - ${total} runs completed, ${fail_count} failed."
echo "=========================================="
