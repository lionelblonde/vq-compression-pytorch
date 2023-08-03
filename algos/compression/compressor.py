import os
from pathlib import Path
import itertools
from contextlib import nullcontext

from tqdm import tqdm

import wandb

import numpy as np
import torch
from torch.nn.utils import clip_grad as cg
from torch.cuda.amp import grad_scaler as gs

from helpers import logger
from helpers.console_util import log_module_info
from algos.compression.autoencoders import VectorQuantizationAutoEncoder


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


class Compressor(object):

    def __init__(self, device, hps):
        self.device = device
        self.hps = hps

        self.iters_so_far = 0
        self.epochs_so_far = 0

        if self.hps.clip_norm <= 0:
            logger.info(f"clip_norm={self.hps.clip_norm} <= 0, hence disabled.")

        self.model = VectorQuantizationAutoEncoder(hps=self.hps).to(self.device)

        self.criteria = self.model.loss_func  # contains several losses

        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.wd,
        )

        self.ctx = (
            torch.amp.autocast(device_type='cuda', dtype=torch.float16 if self.hps.fp16 else torch.float32)
            if self.hps.cuda
            else nullcontext()
        )

        self.scaler = gs.GradScaler(enabled=self.hps.fp16)

        log_module_info(logger, 'compressor_model', self.model)


    def compute_loss(self, x):
        recon_x, vq_loss, perplexity, encoding_indices, normalized_distances = self.model(x)
        losses, table = self.criteria(x, recon_x, vq_loss, encoding_indices, normalized_distances)
        recon_error, loss = losses['recon_error'], losses['loss']
        metrics = {'recon_error': recon_error, 'perplexity': perplexity, 'loss': loss}
        return metrics, loss, table

    def send_to_dash(self, metrics, table=None, mode='unspecified'):
        wandb_dict = {f"{mode}/{k}": v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        wandb.log(wandb_dict, step=self.iters_so_far)
        if table is not None:
            wandb_table = wandb.Table(data=table, columns=["c_idx", "usage(c)"])
            wandb.log({f"{mode}/usage_plot": wandb_table})

    def train(self, train_dataloader, val_dataloader):

        agg_iterable = zip(
            tqdm(train_dataloader),
            itertools.chain.from_iterable(itertools.repeat(val_dataloader)),
            strict=False,
        )

        for i, (t_x, v_x) in enumerate(agg_iterable):

            if self.hps.cuda:
                t_x = t_x.pin_memory().to(self.device, non_blocking=True)
            else:
                t_x = t_x.to(self.device)

            with self.ctx:
                t_metrics, t_loss, _ = self.compute_loss(t_x)
                t_loss /= self.hps.acc_grad_steps

            self.scaler.scale(t_loss).backward()

            if ((i + 1) % self.hps.acc_grad_steps == 0) or (i + 1 == len(train_dataloader)):

                if self.hps.clip_norm > 0:
                    self.scaler.unscale_(self.opt)
                    cg.clip_grad_norm_(self.model.parameters(), self.hps.clip_norm)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                self.send_to_dash(t_metrics, mode='train')
                del t_metrics

            if ((i + 1) % self.hps.eval_every == 0) or (i + 1 == len(train_dataloader)):

                self.model.eval()

                with torch.no_grad():

                    if self.hps.cuda:
                        v_x = v_x.pin_memory().to(self.device, non_blocking=True)
                    else:
                        v_x = v_x.to(self.device)

                    with self.ctx:
                        v_metrics, _, _ = self.compute_loss(v_x)

                    self.send_to_dash(v_metrics, mode='val')
                    del v_metrics

                self.model.train()

            self.iters_so_far += 1

        self.epochs_so_far += 1

    def test(self, dataloader):

        self.model.eval()

        with torch.no_grad():

            for x in tqdm(dataloader):

                if self.hps.cuda:
                    x = x.pin_memory().to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device)

                with self.ctx:
                    metrics, _, table = self.compute_loss(x)

                self.send_to_dash(metrics, table, mode='test')
                del metrics

    def save_to_path(self, path, xtra=None):
        suffix = f"model_{self.epochs_so_far}"
        if xtra is not None:
            suffix += f"_{xtra}"
        suffix += ".tar"
        path = Path(path) / suffix
        torch.save({
            'hps': self.hps,
            'iters_so_far': self.iters_so_far,
            'epochs_so_far': self.epochs_so_far,
            # state_dict's
            'model_state_dict': self.model.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
        }, path)

    def load_from_path(self, path):
        checkpoint = torch.load(path)
        # the "strict" argument of `load_state_dict` is True by default
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['opt_state_dict'])

