import os
import os.path as osp
import itertools

from tqdm import tqdm

import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad as cg
from torch.cuda.amp import grad_scaler as gs
from torch.cuda.amp import autocast_mode as am

from helpers import logger
from helpers.console_util import log_module_info
from algos.classification.models import ClassifierModel


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


class Classifier(object):

    def __init__(self, device, hps, eval_every=100):
        self.device = device
        self.hps = hps

        self.iters_so_far = 0
        self.epochs_so_far = 0

        self.eval_every = eval_every

        if self.hps.clip_norm <= 0:
            logger.info(f"clip_norm={self.hps.clip_norm} <= 0, hence disabled.")

        # Create nets
        self.model = ClassifierModel(
            backbone_name=self.hps.backbone,
            backbone_pretrained=self.hps.pretrained_w_imagenet,
            fc_out_dim=self.hps.num_classes,
        ).to(self.device)

        # Create criterion
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # Set up the optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hps.lr,
        )

        # Set up the gradient scaler for fp16 gpu precision
        self.scaler = gs.GradScaler(enabled=self.hps.fp16)

        # Set up lr scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.opt,
            step_size=40,
            gamma=0.1,
            last_epoch=-1
        )

        log_module_info(logger, 'simclr_model', self.model)

    def compute_loss(self, x, true_y):
        pred_y = self.model(x)
        loss = self.criterion(pred_y, true_y)
        metrics = {'loss': loss}
        return metrics, loss, pred_y

    def send_to_dash(self, metrics, mode='unspecified'):
        wandb_dict = {f"{mode}/{k}": v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        wandb_dict['epoch'] = self.epochs_so_far
        wandb.log(wandb_dict, step=self.iters_so_far)

    def train(self, train_dataloader, val_dataloader):

        samples = 0
        correct = 0

        agg_iterable = zip(tqdm(train_dataloader), itertools.chain.from_iterable(itertools.repeat(val_dataloader)))

        for i, ((t_x, t_true_y), (v_x, v_true_y)) in enumerate(agg_iterable):

            # 1|3>>>> train

            t_x, t_true_y = t_x.to(self.device), t_true_y.to(self.device)
            with am.autocast(enabled=self.hps.fp16):
                t_metrics, t_loss, _ = self.compute_loss(t_x, t_true_y)

            # Update parameters
            self.opt.zero_grad()
            scaled_loss = self.scaler.scale(t_loss)
            scaled_loss.backward()
            if self.hps.clip_norm > 0:
                cg.clip_grad_norm(self.model.parameters(), self.hps.clip_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            # Update lr
            self.scheduler.step()

            self.send_to_dash(t_metrics, mode='train')
            del t_metrics

            # 2|2>>>> evaluate

            if self.iters_so_far % self.eval_every == 0:

                self.model.eval()

                v_x, v_true_y = v_x.to(self.device), v_true_y.to(self.device)
                v_metrics, _, v_pred_y = self.compute_loss(v_x, v_true_y)

                # accuracy

                samples_this_iter = v_true_y.size(0)

                samples += samples_this_iter

                quantized_v_pred_y = (v_pred_y > 0.5).float()
                correct_this_iter = (quantized_v_pred_y == v_true_y).sum().item()
                correct += correct_this_iter

                accuracy_this_iter = correct_this_iter / samples_this_iter
                v_metrics.update({'accuracy': accuracy_this_iter})

                # 3|3>>>> wrap up

                self.send_to_dash(v_metrics, mode='val')
                del v_metrics

            if DEBUG:
                last_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"lr ={last_lr} after {self.iters_so_far} gradient steps")

            self.model.train()

            self.iters_so_far += 1

        self.epochs_so_far += 1

        accuracy_this_epoch = correct / samples
        logger.info(f"accuracy at the end of epoch {self.epochs_so_far} ={accuracy_this_epoch}")

    def test(self, dataloader):

        self.model.eval()

        samples = 0
        correct = 0

        for i, (x, true_y) in enumerate(tqdm(dataloader)):

            x, true_y = x.to(self.device), true_y.to(self.device)
            metrics, _, pred_y = self.compute_loss(x, true_y)

            # accuracy

            samples_this_iter = true_y.size(0)
            samples += samples_this_iter

            quantized_pred_y = (pred_y > 0.5).float()
            correct_this_iter = (quantized_pred_y == true_y).sum().item()
            correct += correct_this_iter

            accuracy_this_iter = correct_this_iter / samples_this_iter
            metrics.update({'accuracy': accuracy_this_iter})

            self.send_to_dash(metrics, mode='test')

        # /!\ training mode is turned back on
        self.model.train()

        accuracy_this_epoch = correct / samples
        logger.info(f"accuracy at the end of testing ={accuracy_this_epoch}")

    def save(self, path, epochs_so_far):
        model_dest_path = osp.join(path, f"model_{epochs_so_far}.tar")
        torch.save(self.model.state_dict(), model_dest_path)

    def load(self, path, epochs_so_far):
        model_orig_path = osp.join(path, f"model_{epochs_so_far}.tar")
        self.model.load_state_dict(torch.load(model_orig_path))
