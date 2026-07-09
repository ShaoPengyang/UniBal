export CUDA_VISIBLE_DEVICES=0,

REPORTTO="wandb"
WANDB_PROJECT="UniBal"

MODEL="Llama-2-7b-hf"
TRAINER="UniSatImp"
TOKENIZER_PRETRAINED="meta-llama/Llama-2-7b-chat-hf"
splits=(
    "News"
)

# UniBal method grid
lr_set=("5e-5")
bz_set=("4 4" "4 8")
epoch_set=(10)
alpha_set=(0.1 1)
gamma_set=(1.0)
beta1_set=(5.0)
beta2_set=(1.0)
sigma_forget_set=(5.0 10.0)
eps_set=(0.14 0.16 0.18)
eta_set=(0.05 0.1)

# Fixed method / probe args
warmup_epochs=0
probe_budget_k=40
probe_build_batch_size=2
probe_eval_batch_size=2

for split in "${splits[@]}"; do
    for lr in "${lr_set[@]}"; do
        for bz in "${bz_set[@]}"; do
            for epochs in "${epoch_set[@]}"; do
                for alpha in "${alpha_set[@]}"; do
                    for gamma in "${gamma_set[@]}"; do
                        for beta1 in "${beta1_set[@]}"; do
                            for beta2 in "${beta2_set[@]}"; do
                                for sigma_forget in "${sigma_forget_set[@]}"; do
                                    for eps in "${eps_set[@]}"; do
                                        for eta in "${eta_set[@]}"; do
                                            # Args ========================================
                                            PRETRAINED_PATH="muse-bench/MUSE-${split}_target"
                                            bsz=$(echo $bz | cut -d' ' -f1)
                                            grad_acc=$(echo $bz | cut -d' ' -f2)

                                            SUFFIX="lr${lr}_b${bsz}_ga${grad_acc}_a${alpha}_g${gamma}_b1${beta1}_b2${beta2}_sigmaF${sigma_forget}_eps${eps}_eta${eta}_e${epochs}"
                                            TASK_NAME="unlearn_muse_${split}_${MODEL}_${TRAINER}_${SUFFIX}"
                                            OUTPUT_DIR="./saves/unlearn/muse/${split}/${MODEL}/${TRAINER}/${SUFFIX}"

                                            # TRAIN COMMAND =================================
                                            export WANDB_PROJECT=${WANDB_PROJECT}
                                            python src/train.py --config-name=unlearn.yaml \
                                                experiment=unlearn/muse/default \
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
                                                trainer.method_args.retain_loss_eps=$eps \
                                                trainer.method_args.dual_step_size=$eta \
                                                trainer.method_args.dual_warmup_epochs=$warmup_epochs \
                                                trainer.method_args.retain_scoring_mode=online \
                                                +trainer.method_args.lambda_update_mode=probe \
                                                +trainer.method_args.probe_budget_k=${probe_budget_k} \
                                                +trainer.method_args.probe_build_batch_size=${probe_build_batch_size} \
                                                +trainer.method_args.probe_eval_batch_size=${probe_eval_batch_size} \
                                                +trainer.method_args.probe_selection_method=kmedoids \
                                                +trainer.method_args.probe_cluster_max_iter=20 \
                                                +trainer.method_args.probe_cache_file=${OUTPUT_DIR}/probe_indices.json \
                                                trainer.method_args.alpha=$alpha \
                                                trainer.method_args.gamma=$gamma \
                                                trainer.method_args.beta1=$beta1 \
                                                trainer.method_args.beta2=$beta2 \
                                                trainer.method_args.retain_loss_type=NLL \
                                                trainer.method_args.sigma_forget=$sigma_forget \
                                                trainer.method_args.sigma_retain=1.0 \
                                                trainer.method_args.forget_dro=True \
                                                trainer.method_args.retain_dro=False \
                                                trainer.method_args.log_ori_loss=True
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
