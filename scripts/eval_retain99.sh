#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

MODEL="Llama-2-7b-chat-hf"
RETAIN_MODEL_PATH="open-unlearning/tofu_Llama-2-7b-chat-hf_retain99"
TASK_NAME="tofu_${MODEL}_retain99"
OUTPUT_DIR="./saves/eval/${TASK_NAME}"

echo "[Eval] Evaluating retain model: ${TASK_NAME} from ${RETAIN_MODEL_PATH}"
python src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=forget01 \
    holdout_split=holdout01 \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    model.model_args.pretrained_model_name_or_path=${RETAIN_MODEL_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${RETAIN_MODEL_PATH} \
    paths.output_dir=${OUTPUT_DIR}

echo "Done. Results saved to ${OUTPUT_DIR}"
