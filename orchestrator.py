import time
import os
import signal
from pathlib import Path

import wandb
import numpy as np

from helpers import logger
from helpers.console_util import timed_cm_wrapper, log_epoch_info
from helpers.dataloader_utils.bigearthnet_utils.dataloader import get_dataloader
from helpers.dataloader_utils.bigearthnet_utils.splitter import split_dataset


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 1)


def learn(
    args,
    algo_wrapper,
    experiment_name,
):

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger, use=DEBUG)

    with timed("splitting"):
        paths_list = split_dataset(args.dataset_handle, num_classes=43)  # 19 is another option here

    with timed("dataloading"):
        dataloaders = []
        tmpdict = {
            "seed": args.seed,
            "data_path": args.data_path,
            "dataset_handle": args.dataset_handle,
            "batch_size": args.batch_size,
            "truncate_at": args.truncate_at,
            "num_workers": args.num_workers,
        }
        # Create the dataloaders
        dataloaders.append(get_dataloader(
            **tmpdict, split_path=paths_list[0], train_stage=True, shuffle=True,
        ))
        dataloaders.append(get_dataloader(
            **tmpdict, split_path=paths_list[1],
        ))
        dataloaders.append(get_dataloader(
            **tmpdict, split_path=paths_list[2],
        ))
        for i, e in enumerate(dataloaders):
            # Log stats about the dataloaders
            ds_len = e.dataset_length
            dl_len = len(e)
            logger.info(
                f"DATALOADER#{i} -|-"
                f"SETLEN={str(ds_len).zfill(7)} -|-"
                f"DLDLEN={str(dl_len).zfill(7)}"
            )

    # Create an algorithm
    algo = algo_wrapper()

    tstart = time.time()

    # Set up model save directory
    ckpt_dir = Path(args.checkpoint_dir) / experiment_name
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    # Save the model as a dry run, to avoid bad surprises at the end
    algo.save_to_path(ckpt_dir, xtra="dryrun")
    logger.info(f"dry run. Saving model @: {ckpt_dir}")

    # Handle timeout signal gracefully
    def timeout(signum, frame):
        # Save the model
        algo.save_to_path(ckpt_dir, xtra="timeout")
        # No need to log a message, orterun stopped the trace already
        # No need to end the run by hand, SIGKILL is sent by orterun fast enough after SIGTERM

    # Tie the timeout handler with the termination signal
    # Note, orterun relays SIGTERM and SIGINT to the workers as SIGTERM signals,
    # quickly followed by a SIGKILL signal (Open-MPI impl)
    signal.signal(signal.SIGTERM, timeout)

    # Group by everything except the seed, which is last, hence index -1
    group = '.'.join(experiment_name.split('.')[:-1])

    # Set up wandb
    while True:
        try:
            wandb.init(
                project=args.wandb_project,
                name=experiment_name,
                id=experiment_name,
                group=group,
                config=args.__dict__,
                dir=args.root,
            )
            break
        except Exception:
            pause = 10
            logger.info(f"wandb co error. Retrying in {pause} secs.")
            time.sleep(pause)
    logger.info("wandb co established!")

    globs = [  # wandb x-axis metrics
        'train',
        'val', 'val-agg',
        'test', 'test-agg',
    ]
    for glob in globs:
        # for each of the globs of metrics, define a custom x-axis
        wandb.define_metric(f"{glob}/step")
        wandb.define_metric(f"{glob}/*", step_metric=f"{glob}/step")

    while algo.epochs_so_far < args.epochs:
        logger.info("training")

        log_epoch_info(logger, algo.epochs_so_far, args.epochs, tstart)

        algo.train(dataloaders[0], dataloaders[1])

        if algo.epochs_so_far % args.save_freq == 0:
            algo.save_to_path(ckpt_dir)

    if algo.epochs_so_far > 0:
        # Save once we are done training
        # (unless we have not done a single epoch of training)
        algo.save_to_path(ckpt_dir, xtra="done")
        logger.info(f"we're done training. Saving model @: {ckpt_dir}.\nbye.")

    logger.info("testing")
    algo.test(dataloaders[2])
