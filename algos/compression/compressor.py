import os
import os.path as osp
import itertools

from tqdm import tqdm

import wandb

import numpy as np
import torch
from torch.nn.utils import clip_grad as cg
from torch.cuda.amp import grad_scaler as gs
from torch.cuda.amp import autocast_mode as am

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

    def __init__(self, device, hps, eval_every=50):
        self.device = device
        self.hps = hps

        self.iters_so_far = 0
        self.epochs_so_far = 0
        self.eval_every = eval_every

        if self.hps.clip_norm <= 0:
            logger.info(f"clip_norm={self.hps.clip_norm} <= 0, hence disabled.")

        # Create nets
        self.model = VectorQuantizationAutoEncoder(hps=self.hps).to(self.device)

        # Create criterion
        self.criteria = self.model.loss_func  # contains several losses

        # Set up the optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.wd,
        )

        # Set up the gradient scaler for fp16 gpu precision
        self.scaler = gs.GradScaler(enabled=self.hps.fp16)

        log_module_info(logger, 'simclr_model', self.model)

    def set_scheduler(self, steps_per_epoch):
        # Set up lr scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            self.hps.max_lr,
            epochs=self.hps.epochs,
            steps_per_epoch=steps_per_epoch,
            final_div_factor=1000,
        )

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

        agg_iterable = zip(tqdm(train_dataloader), itertools.cycle(val_dataloader))
        for i, (t_x, v_x) in enumerate(agg_iterable):

            # 1|3>>>> train

            t_x = t_x.to(self.device)
            with am.autocast(enabled=self.hps.fp16):
                t_metrics, t_loss, _ = self.compute_loss(t_x)

            # Update parameters
            self.opt.zero_grad()
            scaled_loss = self.scaler.scale(t_loss)
            scaled_loss.backward()
            if self.hps.clip_norm > 0:
                cg.clip_grad_norm_(self.model.parameters(), self.hps.clip_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            # Update lr
            self.scheduler.step()

            # 2|2>>>> evaluate

            if self.iters_so_far % self.eval_every == 0:

                self.model.eval()

                v_x = v_x.to(self.device)
                v_metrics, _, _ = self.compute_loss(v_x)

                # 3|3>>>> wrap up
                self.send_to_dash(v_metrics, mode='val')

            self.send_to_dash(t_metrics, mode='train')

            if DEBUG:
                last_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"lr is {last_lr} after {self.iters_so_far} gradient steps")

            self.model.train()

            self.iters_so_far += 1

        self.epochs_so_far += 1

    def test(self, dataloader):
        self.model.eval()

        for i, x in enumerate(tqdm(dataloader)):

            x = x.to(self.device)
            metrics, _, table = self.compute_loss(x)

            self.send_to_dash(metrics, table, mode='test')

        # /!\ training mode is turned back on
        self.model.train()

    def save(self, path, epochs_so_far):
        model_dest_path = osp.join(path, f"model_{epochs_so_far}.tar")
        torch.save(self.model.state_dict(), model_dest_path)

    def load(self, path, epochs_so_far):
        model_orig_path = osp.join(path, f"model_{epochs_so_far}.tar")
        self.model.load_state_dict(torch.load(model_orig_path))
