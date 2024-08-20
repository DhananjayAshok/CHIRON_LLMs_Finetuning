
model_name=roberta
common="--num_train_epochs 5 --learning_rate 1e-4 --model_name_or_path roberta-base"
#model_name=llama3
#common="--num_train_epochs 2 --learning_rate 5e-5 --model_name_or_path eta-llama/Meta-Llama-3.1-8B --peft lora"
data_root=../dataset/intention
python llm-utils/classification_finetune.py --max_seq_len 180 --train_file $data_root/train_m.csv --validation_file $data_root/validation_m.csv --output_dir models/"$model_name"_m $common
python llm-utils/classification_finetune.py --max_seq_len 180 --train_file $data_root/train_ms.csv --validation_file $data_root/validation_ms.csv --output_dir models/"$model_name"_ms $common
python llm-utils/classification_finetune.py --max_seq_len 180 --train_file $data_root/train_msc.csv --validation_file $data_root/validation_msc.csv --output_dir models/"$model_name"_msc $common
#python llm-utils/classification_finetune.py --max_seq_len 180 --train_file $data_root/train_msch.csv --validation_file $data_root/validation_msch.csv --model_name_or_path roberta-large models/"$model_name"_msch $common
