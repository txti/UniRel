export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="0"
export WANDB_MODE="offline"
export INTEL_EXTENSION_FOR_PYTORCH_XPU_DISABLED=1

python3 ./unirel/train.py \
    --task_name UniRel \
    --max_seq_length 100 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 100 \
    --logging_dir ./tb_logs \
    --logging_steps 50 \
    --eval_steps 5000000 \
    --save_steps 5000 \
    --evaluation_strategy steps \
    --warmup_ratio 0.1 \
    --model_dir bert-base-cased \
    --output_dir ./output/nyt \
    --overwrite_output_dir \
    --dataset_dir ./data \
    --dataloader_pin_memory \
    --dataloader_num_workers 4 \
    --lr_scheduler_type cosine \
    --seed 2023 \
    --do_test_all_checkpoints\
    --dataset_name nyt \
    --test_data_type unirel_span \
    --threshold 0.5 \
    --do_train
