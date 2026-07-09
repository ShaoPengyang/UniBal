#!/bin/bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,7}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//,/,}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB__DISABLE_STATS=true

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_SLOTS=${#GPU_IDS[@]}
if [ "$NUM_SLOTS" -lt 1 ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES 解析失败: $CUDA_VISIBLE_DEVICES"
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    for gpu in "${GPU_IDS[@]}"; do
        if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
            echo "ERROR: 非法 GPU 编号 '$gpu'"
            exit 1
        fi
        if [ "$gpu" -ge "$GPU_COUNT" ]; then
            echo "ERROR: GPU '$gpu' 越界。当前仅有 ${GPU_COUNT} 张卡。"
            exit 1
        fi
    done
fi

STAGGER_SEC=60

REPORTTO="wandb"
MODEL="Llama-2-7b-chat-hf"
TRAINER="DrSimNPO"
PRETRAINED_PATH="open-unlearning/tofu_Llama-2-7b-chat-hf_full"
MODEL_TAG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/.' '--')
WANDB_PROJECT_DEFAULT="BalDRO-${MODEL_TAG}-drsimnpo"
export WANDB_PROJECT="${WANDB_PROJECT:-$WANDB_PROJECT_DEFAULT}"

splits=(
    "forget01 holdout01 retain99"
)
lr_set=("5e-5")
bz_set=("8 4")
beta_set=(3.5 4.5)
delta_set=(0 1)
gamma_set=(0.5 1.0)
epoch_set=(15)

run_one() {
    local gpu=$1 split=$2 lr=$3 bz=$4 epochs=$5 beta=$6 delta=$7 gamma=$8 retain_dro=$9 run_idx=${10}
    local forget_split holdout_split retain_split bsz grad_acc
    forget_split=$(echo "$split" | cut -d' ' -f1)
    holdout_split=$(echo "$split" | cut -d' ' -f2)
    retain_split=$(echo "$split" | cut -d' ' -f3)
    bsz=$(echo "$bz" | cut -d' ' -f1)
    grad_acc=$(echo "$bz" | cut -d' ' -f2)

    local SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_b${beta}_d${delta}_g${gamma}_e${epochs}_reDRO${retain_dro}"
    local TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
    local OUTPUT_DIR="./saves/unlearn/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"
    local LOG_FILE="./saves/logs/${TRAINER}_${SUFFIX}_gpu${gpu}.log"

    mkdir -p "$(dirname "$LOG_FILE")"
    echo "=========================================="
    echo "RUN ${run_idx} | GPU ${gpu}: ${SUFFIX}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$gpu python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=${TRAINER} \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=${PRETRAINED_PATH} \
        model.tokenizer_args.pretrained_model_name_or_path=${PRETRAINED_PATH} \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        retain_split=${retain_split} \
        task_name=${TASK_NAME} \
        paths.output_dir="${OUTPUT_DIR}" \
        do_save=False \
        trainer.args.save_strategy=no \
        eval.tofu.retain_logs_path=./saves/eval/tofu_${MODEL}_${retain_split}/TOFU_EVAL.json \
        eval.tofu.batch_size=8 \
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
        trainer.method_args.gamma=$gamma \
        trainer.method_args.alpha=1.0 \
        trainer.method_args.retain_loss_type=NLL \
        trainer.method_args.beta=$beta \
        trainer.method_args.delta=$delta \
        trainer.method_args.sigma_forget=1.0 \
        trainer.method_args.sigma_retain=1.0 \
        trainer.method_args.forget_dro=True \
        trainer.method_args.retain_dro=${retain_dro} \
        trainer.method_args.log_ori_loss=True \
        2>&1 | tee "$LOG_FILE"
}

total=0
fail_count=0
for split in "${splits[@]}"; do
    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            for epochs in "${epoch_set[@]}"; do
                for beta in "${beta_set[@]}"; do
                    for delta in "${delta_set[@]}"; do
                        for gamma in "${gamma_set[@]}"; do
                            for retain_dro in "False" "True"; do
                                gpu=${GPU_IDS[$((total % NUM_SLOTS))]}
                                run_one "$gpu" "$split" "$lr" "$bz" "$epochs" "$beta" "$delta" "$gamma" "$retain_dro" "$((total + 1))" &
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
        done
    done
done

for pid in $(jobs -p); do
    wait "$pid" || fail_count=$((fail_count + 1))
done

echo "=========================================="
echo "ALL DONE — ${total} runs completed, ${fail_count} failed."
echo "=========================================="
