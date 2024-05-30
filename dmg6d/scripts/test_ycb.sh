#!/bin/bash
GPU_NUM=0
GPU_COUNT=1
#NAME='ycbv_swindepose'
#WANDB_PROJ='pose_estimation'
#export CUDA_VISIBLE_DEVICES=$GPU_NUM
#EXP_DIR='ycbv_swindepose/swin_de_pose/train_log'
#LOG_EVAL_DIR="$EXP_DIR/$NAME/ycb/eval_results"
#SAVE_CHECKPOINT="$EXP_DIR/$NAME/ycb/checkpoints"
#LOG_TRAININFO_DIR="$EXP_DIR/$NAME/ycb/train_info"
#tst_mdl="$SAVE_CHECKPOINT/ycb.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 60098 apps/train_ycb.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --lr 1e-2 \
    --dataset_name 'ycb' \
    --data_root 'datasets/ycb/YCB_Video_Dataset' \
    --train_list 'train_data.txt' --test_list 'test_data.txt' \
    --in_c 9 \
    --load_checkpoint train_log/ycb/checkpoints/FFB6D_ycb.pth.tar \
    --test --test_pose --eval_net --test_gt \
    --mini_batch_size 8 --val_mini_batch_size 8 --test_mini_batch_size 1 \
    --log_eval_dir ycbv_swindepose/swin_de_pose/train_log/ycb/eval_results --save_checkpoint ycbv_swindepose/swin_de_pose/train_log/ycb/checkpoints --log_traininfo_dir ycbv_swindepose/swin_de_pose/train_log/ycb/train_info