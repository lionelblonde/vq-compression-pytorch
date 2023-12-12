#!/usr/bin/env bash

python main.py \
    --seed 0 \
    --cuda \
    --fp16 \
    --wandb_project ada \
    --dataset_handle bigearthnet \
    --epoch 20 \
    --batch_size 128 \
    --save_freq 1 \
    --eval_every 8 \
    --lr 1e-3 \
    --wd 0 \
    --clip_norm 0 \
    --algo_handle 'residualvqae' \
    --data_path /hdd/datasets/BigEarthNet-v1.0 \
    --truncate_at 100 \
    --acc_grad_steps 8 \
    --num_workers 4 \
    --num_quantizers 8 \
    --quantize_dropout \
    --codebook_size 100 \
    --kmeans_iters 100 \
    --threshold_ema_dead_code 2 \
    --no-learnable_codebook

