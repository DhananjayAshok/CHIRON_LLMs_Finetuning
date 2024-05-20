export DATA_NAME=no_board_history_with_sys_history_cicero
export LEARNING_RATE=5e-5
export BATCH_SIZE=2
export EPOCH=10
export DATA_PATH=../dataset/${DATA_NAME}.json
export CKPT_PATH=meta-llama/Llama-2-7b-chat-hf
export SAVE_NAME=no_board_history_with_sys_history_cicero_lr${LEARNING_RATE}_batch${BATCH_SIZE}_epoch${EPOCH}
export SAVE_PATH=../output/${SAVE_NAME}


python sft.py \
    --dataset_name=${DATA_PATH}\
    --model_name_or_path=${CKPT_PATH} \
    --report_to="wandb" \
    --learning_rate=${LEARNING_RATE} \
    --per_device_train_batch_size=${BATCH_SIZE} \
    --gradient_accumulation_steps=16 \
    --output_dir=${SAVE_PATH} \
    --logging_steps=1 \
    --num_train_epochs=${EPOCH} \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "v_proj"\
    --torch_dtype=bfloat16 \
    --bf16=True \
    2>&1 | tee ${SAVE_NAME}.log