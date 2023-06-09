#!/usr/bin/env bash

export DATASET_DIR=/Users/lionelblonde/Documents/datasets

python main.py \
    --no-cuda \
    --wandb_project testSSL \
    --dataset_handle bigearthnet\
    --val_split 0.15\
    --test_split 0.15\
    --epoch 100\
    --batch_size 128\
    --save_freq 1\
    --lr 3e-4\
    --wd 1e-6\
    --clip_norm 60.\
    --algo_handle 'bigearthnet_classifier'\
    --backbone 'resnet18'\
    --pretrained_w_imagenet\
    --fc_hid_dim 128 \
    --fc_out_dim 64 \
    --finetune_probe_epochs 50 \
    --finetune_probe_batch_size 256 \
    --task train \
    --data_path /Users/lionelblonde/Datasets/BigEarthNet-v1.0 \
    --truncate_at 5000
