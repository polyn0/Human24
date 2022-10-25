#sketchy module
export DATA_DIR=dataset
export TASK_NAME=squad
python run_cls.py \
    --model_type koelectra \
    --model_name_or_path monologg/koelectra-base-v3-discriminator \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 512 \
    --per_gpu_train_batch_size=6   \
    --per_gpu_eval_batch_size=8   \
    --warmup_steps=814 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --eval_all_checkpoints \
    --output_dir squad/cls_squad2_koelectra-base \
    --save_steps 2500 \
    --fp16