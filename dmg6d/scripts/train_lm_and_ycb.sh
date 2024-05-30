#!/bin/bash
#
#CLS='ape'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='benchvise'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='cam'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='can'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='cat'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='driller'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='duck'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='eggbox'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='glue'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='holepuncher'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='iron'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='lamp'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130
#
#CLS='phone'
#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 4 \
#    --gpu_id 1 \
#    --gpu '1' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'test.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --mini_batch_size 6 --val_mini_batch_size 6 \
#    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
#    --n_total_epoch=130

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60088 apps/train_ycb.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --lr 1e-2 \
    --dataset_name 'ycb' \
    --data_root 'datasets/ycb/YCB_Video_Dataset' \
    --train_list 'train_data_list_20230706.txt' --test_list 'test_data_list_20230706.txt' \
    --in_c 9 \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir train_log/ycb/eval_results --save_checkpoint train_log/ycb/checkpoints --log_traininfo_dir train_log/ycb/train_info \
    --n_total_epoch=150
