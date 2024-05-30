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

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60008 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --load_checkpoint occlm/occlm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
    --train_list "train_pbr/training_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --test --test_pose --eval_net \
    --mini_batch_size 9 --val_mini_batch_size 9 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info

CLS='can'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60008 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --load_checkpoint occlm/occlm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
    --train_list "train_pbr/training_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --test --test_pose --eval_net \
    --mini_batch_size 9 --val_mini_batch_size 9 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info

CLS='cat'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60008 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --load_checkpoint occlm/occlm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
    --train_list "train_pbr/training_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --test --test_pose --eval_net \
    --mini_batch_size 9 --val_mini_batch_size 9 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info

CLS='driller'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60008 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --load_checkpoint occlm/occlm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
    --train_list "train_pbr/training_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --test --test_pose --eval_net \
    --mini_batch_size 9 --val_mini_batch_size 9 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info

CLS='duck'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60008 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --load_checkpoint occlm/occlm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
    --train_list "train_pbr/training_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --test --test_pose --eval_net \
    --mini_batch_size 9 --val_mini_batch_size 9 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info

CLS='eggbox'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60008 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --load_checkpoint occlm/occlm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
    --train_list "train_pbr/training_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --test --test_pose --eval_net \
    --mini_batch_size 9 --val_mini_batch_size 9 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info

CLS='glue'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60008 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --load_checkpoint occlm/occlm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
    --train_list "train_pbr/training_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --test --test_pose --eval_net \
    --mini_batch_size 9 --val_mini_batch_size 9 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info

CLS='holepuncher'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60008 apps/train_occlm.py \
    --gpus=2 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'occlusion_linemod' \
    --data_root 'datasets/occ_linemod/Occ_LineMod' \
    --load_checkpoint occlm/occlm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
    --train_list "train_pbr/training_msk_"$CLS".txt" --test_list "test/testing_msk_"$CLS".txt" \
    --occ_linemod_cls=$CLS \
    --in_c 9 \
    --test --test_pose --eval_net \
    --mini_batch_size 9 --val_mini_batch_size 9 --test_mini_batch_size 1 \
    --log_eval_dir occlm/train_log/$CLS/eval_results --save_checkpoint occlm/train_log/$CLS/checkpoints --log_traininfo_dir occlm/train_log/$CLS/train_info
