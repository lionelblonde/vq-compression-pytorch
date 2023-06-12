from typing import Optional

import argparse


def boolean_flag(
    parser: argparse.ArgumentParser,
    name: str,
    default: Optional[bool] = False,
    help: Optional[str] = None,
):
    """Add a boolean flag to argparse parser."""
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def agg_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # meta
    parser.add_argument("--task", type=str, choices=['train'], default=None)  # XXX: adding eval or inference soon
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    parser.add_argument("--uuid", type=str, default=None)
    # resources
    boolean_flag(parser, "cuda", default=True, help="whether to use gpu(s) or cpu(s)")
    boolean_flag(parser, "fp16", default=False, help="whether to use fp16 precision")
    # logging
    parser.add_argument("--wandb_project", help="wandb project name", default='DEFAULT')
    # dataset
    parser.add_argument("--dataset_handle", type=str, default=None)
    parser.add_argument("--data_path", type=str, help="path to folders with images to train on.")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--truncate_at", type=int, default=None, help="amount of data to keep")
    # training
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train model")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for SSL")
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=20)
    # opt
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--clip_norm", type=float, default=60.)
    # algo
    parser.add_argument("--algo_handle", type=str, choices=[
        'bigearthnet_classifier',
        'bigearthnet_simclr',
        'bigearthnet_compressor',
    ], default=None)
    parser.add_argument("--env_id", type=str, default='')

    # >>>> classifier

    # model architecture
    parser.add_argument("--backbone", type=str, default=None)
    boolean_flag(parser, "pretrained_w_imagenet", default=False)
    parser.add_argument("--fc_hid_dim", type=int, default=128)

    # >>>> simclr

    # model architecture
    # >>>>>>>> overlap w/ classifier
    # parser.add_argument("--backbone", type=str, choices=['resnet18', 'resnet50'], default=None)
    # boolean_flag(parser, "pretrained_w_imagenet", default=True)
    # parser.add_argument("--fc_hid_dim", type=int, default=128)
    # <<<<<<<<
    parser.add_argument("--fc_out_dim", type=int, default=64)
    # fine-tuning or linear probing
    boolean_flag(parser, "linear_probe", default=False)
    boolean_flag(parser, "fine_tuning", default=False)
    parser.add_argument("--finetune_probe_epochs", type=int, default=10)  # same as SimCLR
    parser.add_argument("--finetune_probe_batch_size", type=int, default=256)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # >>>> compressor

    # opt 2
    parser.add_argument("--max_lr", type=float, default=1e-3, help="max lr for OneCycleLR scheduler")
    # model architecture
    parser.add_argument("--in_channels", type=int, default=3, help="input channels")
    parser.add_argument("--z_channels", type=int, default=32, help="channels in the latent z")
    parser.add_argument("--ae_hidden", type=int, default=128, help="channels in hid layers in enc/dec.")
    parser.add_argument("--ae_resblocks", type=int, default=2, help="resblocks in enc/dec")
    parser.add_argument("--ae_kernel", type=int, default=4, help="size of kernel in down/up-sampling layers")
    parser.add_argument("--dsf", type=int, choices=[8, 4, 2, 1], default=8, help="downsampling factor (1=no downsampling)")
    # loss
    parser.add_argument("--alpha", type=float, default=1., help="weight of soft entropy loss in total")
    parser.add_argument("--beta", type=float, default=1., help="weight of entropy loss in total")
    # centers
    parser.add_argument("--c_num", type=int, default=8, help="centers")
    parser.add_argument("--c_min", type=float, default=-2., help="initial min value of centers")
    parser.add_argument("--c_max", type=float, default=2., help="initial max value of centers")

    return parser
