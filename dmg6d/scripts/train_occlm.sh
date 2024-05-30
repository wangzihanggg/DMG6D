#!/bin/bash

#    'ape':1,
#    'can':5,
#    'cat':6,
#    'driller':8,
#    'duck':9,
#    'eggbox':10,
#    'glue':11,
#    'holepuncher':12

CLS='ape'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60076 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 --cyc_max_lr 1e-3 --cyc_base_lr 1e-5 --clr_div 9 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --train_list "test/testing_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --mini_batch_size 6 --val_mini_batch_size 6 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info \
    --n_total_epoch=120

CLS='can'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60076 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 --cyc_max_lr 1e-3 --cyc_base_lr 1e-5 --clr_div 9 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --train_list "test/testing_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --mini_batch_size 6 --val_mini_batch_size 6 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info \
    --n_total_epoch=120

CLS='cat'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60076 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 --cyc_max_lr 1e-3 --cyc_base_lr 1e-5 --clr_div 9 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --train_list "test/testing_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --mini_batch_size 6 --val_mini_batch_size 6 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info \
    --n_total_epoch=120

CLS='driller'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60076 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 --cyc_max_lr 1e-3 --cyc_base_lr 1e-5 --clr_div 9 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --train_list "test/testing_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --mini_batch_size 6 --val_mini_batch_size 6 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info \
    --n_total_epoch=120

CLS='duck'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60076 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 --cyc_max_lr 1e-3 --cyc_base_lr 1e-5 --clr_div 9 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --train_list "test/testing_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --mini_batch_size 6 --val_mini_batch_size 6 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info \
    --n_total_epoch=120

CLS='eggbox'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60076 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 --cyc_max_lr 1e-3 --cyc_base_lr 1e-5 --clr_div 9 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --train_list "test/testing_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --mini_batch_size 6 --val_mini_batch_size 6 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info \
    --n_total_epoch=120

CLS='glue'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60076 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 --cyc_max_lr 1e-3 --cyc_base_lr 1e-5 --clr_div 9 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --train_list "test/testing_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --mini_batch_size 6 --val_mini_batch_size 6 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info \
    --n_total_epoch=120

CLS='holepuncher'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60076 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 --cyc_max_lr 1e-3 --cyc_base_lr 1e-5 --clr_div 9 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --train_list "test/testing_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --mini_batch_size 6 --val_mini_batch_size 6 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info \
    --n_total_epoch=120
