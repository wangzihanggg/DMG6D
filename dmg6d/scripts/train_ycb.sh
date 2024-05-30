#!/bin/bash
GPU_NUM=0
GPU_COUNT=1
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 60088 apps/train_ycb.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --lr 1e-2 \
    --dataset_name 'ycb' \
    --data_root 'datasets/ycb/YCB_Video_Dataset' \
    --train_list 'train_data_list_20230706.txt' --test_list 'test_data_list_20230706.txt' \
    --in_c 9 \
    --load_checkpoint train_log/ycb/checkpoints/FFB6D_ycb.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir train_log/ycb/eval_results --save_checkpoint train_log/ycb/checkpoints --log_traininfo_dir train_log/ycb/train_info \
    --n_total_epoch=25