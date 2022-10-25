#intensive module
export SQUAD_DIR=dataset /
python run_squad_av.py \
    --model_type koelectra \
    --model_name_or_path monologg/koelectra-base-v3-discriminator \
    --do_train \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_train_batch_size=6 \
    --per_gpu_eval_batch_size=8 \
    --warmup_steps=814 \
    --output_dir squad/squad2_koelectra-base \
    --eval_all_checkpoints \
    --save_steps 2500 \
    --n_best_size=20 \
    --max_answer_length=30 \
    --fp16
