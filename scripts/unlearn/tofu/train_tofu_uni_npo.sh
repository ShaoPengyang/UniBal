export CUDA_VISIBLE_DEVICES=0,

REPORTTO="wandb"
WANDB_PROJECT="UniBal"

MODEL="Llama-2-7b-chat-hf"
TRAINER="UniNPO"
splits=(
    "forget01 holdout01 retain99"
)
PRETRAINED_PATH="open-unlearning/tofu_Llama-2-7b-chat-hf_full"

# UniBal method grid
eps_set=(0.16 0.18)
eta_set=(0.05)
beta_dv_set=(1.0 2.0)
alpha_set=(1.0)
lr_set=("5e-5")

# Fixed training / probe args
bsz=8
grad_acc=2
epochs=15
beta_npo=0.1
gamma=1.0
top_k=160
warmup_epochs=0
probe_budget_k=40
probe_build_batch_size=10
probe_eval_batch_size=10

for split in "${splits[@]}"; do
    for lr in "${lr_set[@]}"; do
        for eps in "${eps_set[@]}"; do
            for eta in "${eta_set[@]}"; do
                for beta_dv_forget in "${beta_dv_set[@]}"; do
                    for alpha in "${alpha_set[@]}"; do
                        # Args ========================================
                        forget_split=$(echo $split | cut -d' ' -f1)
                        holdout_split=$(echo $split | cut -d' ' -f2)
                        retain_split=$(echo $split | cut -d' ' -f3)

                        SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_eps${eps}_eta${eta}_betaDV${beta_dv_forget}_a${alpha}_e${epochs}"
                        TASK_NAME="unlearn_tofu_${MODEL}_${forget_split}_${TRAINER}_${SUFFIX}"
                        OUTPUT_DIR="./saves/unlearn/tofu/${forget_split}/${MODEL}/${TRAINER}/${SUFFIX}"

                        # TRAIN COMMAND =================================
                        export WANDB_PROJECT=${WANDB_PROJECT}
                        python src/train.py --config-name=unlearn.yaml \
                            experiment=unlearn/tofu/default \
                            collator=DataCollatorForSupervisedDatasetwithIndex \
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
                            trainer.method_args.retain_loss_eps=$eps \
                            trainer.method_args.dual_step_size=$eta \
                            trainer.method_args.top_k_retain=$top_k \
                            trainer.method_args.dual_warmup_epochs=$warmup_epochs \
                            trainer.method_args.retain_eval_batch_size=10 \
                            trainer.method_args.retain_eval_ratio=1.0 \
                            trainer.method_args.retain_eval_interval=1 \
                            trainer.method_args.retain_scoring_mode=online \
                            trainer.method_args.retain_score_ema=0.2 \
                            +trainer.method_args.retain_lambda_use_topk=True \
                            +trainer.method_args.lambda_update_mode=probe \
                            +trainer.method_args.probe_budget_k=${probe_budget_k} \
                            +trainer.method_args.probe_build_batch_size=${probe_build_batch_size} \
                            +trainer.method_args.probe_eval_batch_size=${probe_eval_batch_size} \
                            +trainer.method_args.probe_selection_method=kmedoids \
                            +trainer.method_args.probe_cache_file=${OUTPUT_DIR}/probe_indices.json \
                            trainer.method_args.beta=$beta_npo \
                            trainer.method_args.gamma=$gamma \
                            trainer.method_args.alpha=$alpha \
                            trainer.method_args.retain_loss_type=NLL \
                            trainer.method_args.beta_dv_forget=$beta_dv_forget \
                            trainer.method_args.beta_dv_retain=1.0 \
                            trainer.method_args.forget_dro=True \
                            trainer.method_args.retain_dro=False
                    done
                done
            done
        done
    done
done
