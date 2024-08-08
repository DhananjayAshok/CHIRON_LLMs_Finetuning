common="--num_train_epochs 5 --learning_rate 1e-4 --model_name_or_path roberta-large"
data_root=../dataset/intention
python classification_finetune.py --max_seq_len 180 --train_file $data_root/train_m.csv --validation_file $data_root/validation_m.csv --output_dir models/int_clf_m $common