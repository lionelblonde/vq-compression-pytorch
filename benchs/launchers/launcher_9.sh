#!/usr/bin/env bash

export DATASET_DIR=/scratch

export MODEL_DIR=/srv/beegfs/scratch/shares/dmml/eo4eu/datasets/model

python spawner.py \
    --config benchs/configs/very_few_labels/finetune.yaml \
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
