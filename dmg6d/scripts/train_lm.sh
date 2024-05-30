#!/bin/bash

CLS='ape'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='benchvise'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='cam'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='can'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='cat'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='driller'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='duck'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='eggbox'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='glue'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='holepuncher'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='iron'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='lamp'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40

CLS='phone'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 1 \
    --gpu '1' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --mini_batch_size 6 --val_mini_batch_size 6 \
    --log_eval_dir "train_log/$CLS/eval_results" --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info" \
    --n_total_epoch=40
