#!/bin/bash

CLS='ape'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='benchvise'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='cam'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='can'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='cat'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='driller'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='duck'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='eggbox'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='glue'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='holepuncher'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='iron'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='lamp'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

CLS='phone'

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
    --gpus=2 \
    --num_threads 0 \
    --gpu_id 0 \
    --gpus 2 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint train_log/$CLS/checkpoints/$CLS.pth.tar \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 60018 apps/train_lm.py \
#    --gpus=2 \
#    --num_threads 0 \
#    --gpu_id 0 \
#    --gpus 2 \
#    --gpu '0' \
#    --lr 1e-2 \
#    --dataset_name 'linemod' \
#    --data_root 'datasets/linemod/Linemod_preprocessed' \
#    --train_list 'train.txt' --test_list 'test.txt' \
#    --linemod_cls=$CLS \
#    --in_c 9 --lm_no_pbr \
#    --load_checkpoint train_log/linemod/lm_swinTiny_"$CLS"_fullSyn_dense_fullInc/$CLS/checkpoints/FFB6D_$CLS.pth.tar \
#    --test --test_pose --eval_net \
#    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
#    --log_eval_dir train_log/$CLS/eval_results --save_checkpoint "train_log/$CLS/checkpoints" --log_traininfo_dir "train_log/$CLS/train_info"
