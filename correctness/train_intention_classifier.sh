model_name=llama3
common="--num_train_epochs 2 --learning_rate 5e-5 --model_name_or_path eta-llama/Meta-Llama-3.1-8B --peft lora"
data_root=../dataset/intention
python classification_finetune.py --max_seq_len 180 --train_file $data_root/train_og_score.csv --validation_file $data_root/validation_og_score.csv --output_dir models/"$model_name"_og_score $common