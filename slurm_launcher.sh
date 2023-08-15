#!/usr/bin/env bash

# export DATASET_DIR=/srv/beegfs/scratch/shares/dmml/eo4eu/datasets
export DATASET_DIR=/share/users/${USER:0:1}/${USER}
# export MODEL_DIR=/home/users/b/blondeli/Code/geo-pytorch/data/checkpoints/weepoo_zoofer_shoophu.gitSHA_81c7d86.simclr_1.seed00
# export MODEL_DIR=/home/users/b/blondeli/Code/geo-pytorch/data/checkpoints/booshe_fapoo_hoozu.gitSHA_cf6299e.simclr_1.seed00
# export MODEL_DIR=/home/users/b/blondeli/Code/geo-pytorch/data/checkpoints/serchoo_huze_woochoo.gitSHA_50f8ca5.simclr_1.seed00
# export MODEL_DIR=/home/users/b/blondeli/Code/geo-pytorch/data/checkpoints/pheetu_zuhoo_chooher.gitSHA_9653a1e.simclr_1.seed00
export MODEL_DIR=/home/users/b/blondeli/Code/geo-pytorch/data/checkpoints/sheepa_sashu_soosu.gitSHA_a9a0107.simclr_1.seed00
# the latter var is not always used but need be here for some configs

python spawner.py \
    --config configs/ssl/linear_probing.yaml \
    --conda_env geocuda \
    --deployment slurm \
    --num_seeds 1 \
    --caliber veryverylong \
    --deploy_now \
    --no-sweep \
    --no-wandb_upgrade \
    --no-wandb_dryrun \
    --debug \
    --debug_lvl 0 \
    --no-quick
