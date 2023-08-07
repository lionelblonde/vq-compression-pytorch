import os
from pathlib import Path
import itertools
from contextlib import nullcontext

from tqdm import tqdm

import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad as cg
from torch.cuda.amp import grad_scaler as gs

from helpers import logger
from helpers.console_util import log_module_info
from helpers.metrics_util import compute_metrics, MetricsAggregator
from algos.classification.models import ClassifierModelTenChan


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


class Classifier(object):

    def __init__(self, device, hps):
        self.device = device
        self.hps = hps

        self.iters_so_far = 0
        self.epochs_so_far = 0

        if self.hps.clip_norm <= 0:
            logger.info(f"clip_norm={self.hps.clip_norm} <= 0, hence disabled.")

        self.model = ClassifierModelTenChan(
            backbone_name=self.hps.backbone,
            backbone_pretrained=self.hps.pretrained_w_imagenet,
            fc_out_dim=self.hps.num_classes,
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hps.lr,
        )

        self.ctx = (
            torch.amp.autocast(
                device_type='cuda',
                dtype=torch.float16 if self.hps.fp16 else torch.float32,
            )
            if self.hps.cuda
            else nullcontext()
        )

        self.scaler = gs.GradScaler(enabled=self.hps.fp16)

        self.metrics = MetricsAggregator(
            self.hps.num_classes,
            self.hps.batch_size,
        )

        log_module_info(logger, 'classifier_model', self.model)

    def compute_loss(self, x, true_y):
        pred_y = self.model(x)
        loss = self.criterion(pred_y, true_y)
        metrics = {'loss': loss}
        return metrics, loss, pred_y

    def send_to_dash(self, metrics, step=None, mode='unspecified'):
        wandb_dict = {
            f"{mode}/{k}": v.item() if hasattr(v, 'item') else v
            for k, v in metrics.items()
        }
        if mode == 'val':
            wandb_dict['epoch'] = self.epochs_so_far
            logger.info("epoch sent to wandb")
        if step is None:
            step = self.iters_so_far  # use iters in x-axis by default
            logger.warn("arg step unspecified; set to iter by default")
        wandb.log(wandb_dict, step=step)

    def train(self, train_dataloader, val_dataloader):

        agg_iterable = zip(
            tqdm(train_dataloader),
            itertools.chain.from_iterable(itertools.repeat(val_dataloader)),
            strict=False,
        )
        balances = torch.Tensor(val_dataloader.balances)
        if self.hps.cuda:
            balances = balances.pin_memory().to(self.device, non_blocking=True)
        else:
            balances = balances.to(self.device)

        for i, ((t_x, t_true_y), (v_x, v_true_y)) in enumerate(agg_iterable):

            if self.hps.cuda:
                t_x = t_x.pin_memory().to(self.device, non_blocking=True)
                t_true_y = t_true_y.pin_memory().to(self.device, non_blocking=True)
            else:
                t_x, t_true_y = t_x.to(self.device), t_true_y.to(self.device)

            with self.ctx:
                t_metrics, t_loss, _ = self.compute_loss(t_x, t_true_y)
                t_loss /= self.hps.acc_grad_steps

            self.scaler.scale(t_loss).backward()

            if ((i + 1) % self.hps.acc_grad_steps == 0) or (i + 1 == len(train_dataloader)):

                if self.hps.clip_norm > 0:
                    self.scaler.unscale_(self.opt)
                    cg.clip_grad_norm_(self.model.parameters(), self.hps.clip_norm)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                self.send_to_dash(t_metrics, step=self.iters_so_far, mode='train')
                del t_metrics

            if ((i + 1) % self.hps.eval_every == 0) or (i + 1 == len(train_dataloader)):

                self.model.eval()

                with torch.no_grad():

                    if self.hps.cuda:
                        v_x = v_x.pin_memory().to(self.device, non_blocking=True)
                        v_true_y = v_true_y.pin_memory().to(self.device, non_blocking=True)
                    else:
                        v_x, v_true_y = v_x.to(self.device), v_true_y.to(self.device)

                    with self.ctx:
                        v_metrics, _, v_pred_y = self.compute_loss(v_x, v_true_y)
                        # compute evaluation scores
                        v_pred_y = (v_pred_y >= 0.).long()
                        v_metrics.update(compute_metrics(
                            v_pred_y, v_true_y,
                            weights=balances,
                        ))
                        self.metrics.step(v_pred_y, v_true_y)

                    self.send_to_dash(v_metrics, step=self.iters_so_far, mode='val')
                    del v_metrics

                self.model.train()

            self.iters_so_far += 1

        self.send_to_dash(self.metrics.compute(), step=self.epochs_so_far, mode='val-agg')
        self.metrics.reset()
        self.epochs_so_far += 1

    def test(self, dataloader):

        balances = torch.Tensor(dataloader.balances)
        if self.hps.cuda:
            balances = balances.pin_memory().to(self.device, non_blocking=True)
        else:
            balances = balances.to(self.device)

        self.model.eval()

        with torch.no_grad():

            for i, (x, true_y) in enumerate(tqdm(dataloader)):

                if self.hps.cuda:
                    x = x.pin_memory().to(self.device, non_blocking=True)
                    true_y = true_y.pin_memory().to(self.device, non_blocking=True)
                else:
                    x, true_y = x.to(self.device), true_y.to(self.device)

                with self.ctx:
                    metrics, _, pred_y = self.compute_loss(x, true_y)
                    # compute evaluation scores
                    pred_y = (pred_y >= 0.).long()
                    metrics.update(compute_metrics(
                        pred_y, true_y,
                        weights=balances,
                    ))
                    self.metrics.step(pred_y, true_y)

                self.send_to_dash(metrics, step=i, mode='test')
                del metrics

        self.send_to_dash(self.metrics.compute(), step=i, mode='test-agg')
        # use `i` from previous loop to see over how many steps the stats are aggregated

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
        if 'iters_so_far' in checkpoint:
            self.iters_so_far = checkpoint['iters_so_far']
        if 'epochs_so_far' in checkpoint:
            self.epochs_so_far = checkpoint['epochs_so_far']
        # the "strict" argument of `load_state_dict` is True by default
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['opt_state_dict'])

