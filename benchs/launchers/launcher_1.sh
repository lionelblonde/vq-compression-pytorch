#!/usr/bin/env bash

export DATASET_DIR=/scratch

python spawner.py \
    --config benchs/configs/few_labels/classifier.yaml \
    --conda_env geocuda \
    --deployment slurm \
    --num_seeds 1 \
    --caliber long \
    --deploy_now \
    --no-sweep \
    --no-wandb_upgrade \
    --no-wandb_dryrun \
    --debug \
    --debug_lvl 0
