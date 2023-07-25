#!/usr/bin/env bash

python main.py \
    --seed 0 \
    --cuda \
    --fp16 \
    --wandb_project pikachu \
    --dataset_handle bigearthnet \
    --val_split 0.25 \
    --test_split 0.25 \
    --epoch 100 \
    --batch_size 128 \
    --save_freq 1 \
    --eval_every 1 \
    --lr 1e-3 \
    --wd 1e-4 \
    --clip_norm 0 \
    --algo_handle 'simclr' \
    --no-linear_probe \
    --fc_hid_dim 128 \
    --backbone resnet18 \
    --no-pretrained_w_imagenet \
    --fc_hid_dim 128 \
    --finetune_probe_epochs 50 \
    --finetune_probe_batch_size 256 \
    --task train \
    --data_path /hdd/datasets/BigEarthNet-v1.0 \
    --truncate_at 10 \
    --num_workers 4 #\
    #--load_checkpoint /hdd/models/SIMCLR_MODELS/phupu_weerer_ceeree-simclr-3epochs/model_3_done.tar
