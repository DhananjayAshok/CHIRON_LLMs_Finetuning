export DATA_PATH=../dataset/data_with_board_history.json
export CKPT_PATH=meta-llama/Llama-2-7b-chat-hf
export SAVE_PATH=../output/with_board_only_history


python sft.py \
    --dataset_name=${DATA_PATH}\
    --model_name_or_path=${CKPT_PATH} \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=16 \
    --output_dir=${SAVE_PATH} \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16 \
    --load_in_8bit
    