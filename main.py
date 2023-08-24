import os
from pathlib import Path

import torch
from torch.backends import cudnn as cudnn

import orchestrator
from helpers import logger
from helpers.argparser_util import agg_argparser
from helpers.experiment import ExperimentInitializer
from algos.classification.classifier import Classifier
from algos.ssl.simclr import SimCLR
from algos.compression.compressor import Compressor


def run(args):

    # Initialize and configure experiment
    experiment = ExperimentInitializer(args)
    experiment.configure_logging()

    # Create experiment name
    experiment_name = experiment.get_name()

    # Set device-related knobs
    assert not args.fp16 or args.cuda, "fp16 ==> cuda"
    if args.cuda:
        # Use cuda
        assert torch.cuda.is_available()
        cudnn.benchmark = False
        cudnn.deterministic = True
        device = torch.device("cuda:0")
    else:
        if torch.has_mps:
            # Use Apple's Metal Performance Shaders (MPS)
            device = torch.device("mps")
        else:
            # Default case: just use plain old cpu, no cuda or m-chip gpu
            device = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage
    args.device = device  # add the device to hps for convenience
    logger.info(f"device in use: {device}")

    # Seedify
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset_handle == 'bigearthnet':
        args.num_classes = 1  # original classes: 43, reduced: 19; 1 if class #2 only (in 43 list)

    if args.algo_handle == 'classifier':
        algo_class_handle = Classifier
        num_transforms = 1
        with_labels = True
        knn_eval = False
    elif args.algo_handle == 'simclr':
        algo_class_handle = SimCLR
        num_transforms = 2
        with_labels = True
        knn_eval = True
    elif args.algo_handle == 'compressor':
        algo_class_handle = Compressor
        num_transforms = 1
        with_labels = False
        knn_eval = False
    else:
        raise NotImplementedError("algorithm not covered")

    def algo_wrapper():
        return algo_class_handle(
            device=device,
            hps=args,
        )

    # Train
    orchestrator.learn(
        args=args,
        algo_wrapper=algo_wrapper,
        experiment_name=experiment_name,
        num_transforms=num_transforms,
        with_labels=with_labels,
        knn_eval=knn_eval,
    )


if __name__ == '__main__':

    _args = agg_argparser().parse_args()

    _args.root = Path(__file__).resolve().parent  # make the paths absolute
    for k in ['checkpoints', 'logs']:
        new_k = f"{k[:-1]}_dir"
        vars(_args)[new_k] = Path(_args.root) / 'data' / k

    run(_args)

