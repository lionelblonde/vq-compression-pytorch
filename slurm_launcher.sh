#!/usr/bin/env bash

# export DATASET_DIR=/srv/beegfs/scratch/shares/dmml/eo4eu/datasets
export DATASET_DIR=/share/users/${USER:0:1}/${USER}
export MODEL_DIR=/home/users/b/blondeli/Code/geo-pytorch/data/checkpoints/weepoo_zoofer_shoophu.gitSHA_81c7d86.simclr_1.seed00
# the latter var is not always used but need be here for some configs

python spawner.py \
    --config configs/ssl/simclr.yaml \
    --conda_env geocuda \
    --deployment slurm \
    --num_seeds 2 \
    --caliber veryverylong \
    --deploy_now \
    --no-sweep \
    --no-wandb_upgrade \
    --no-wandb_dryrun \
    --debug \
    --debug_lvl 0 \
    --no-quick
