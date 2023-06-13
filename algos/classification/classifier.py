import os
import os.path as osp
import itertools

from tqdm import tqdm

import wandb

import numpy as np
import sklearn.metrics as skm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad as cg
from torch.cuda.amp import grad_scaler as gs
from torch.cuda.amp import autocast_mode as am

from helpers import logger
from helpers.console_util import log_module_info
from algos.classification.models import ClassifierModel
# from algos.classification.models import PaperClassifierModel


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

        # Create nets
        # self.model = PaperClassifierModel(
        #     fc_hid_dim=self.hps.fc_hid_dim,
        #     fc_out_dim=self.hps.num_classes,
        # ).to(self.device)
        self.model = ClassifierModel(
            backbone_name=self.hps.backbone,
            backbone_pretrained=self.hps.pretrained_w_imagenet,
            fc_out_dim=self.hps.num_classes,
        ).to(self.device)

        # Create criterion
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

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

        # samples = 0
        # correct = 0

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
                cg.clip_grad_norm_(self.model.parameters(), self.hps.clip_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            # Update lr
            # self.scheduler.step()  # FIXME

            self.send_to_dash(t_metrics, mode='train')
            del t_metrics

            # 2|2>>>> evaluate

            if self.iters_so_far % self.hps.eval_every == 0:

                self.model.eval()

                v_x, v_true_y = v_x.to(self.device), v_true_y.to(self.device)
                v_metrics, _, v_pred_y = self.compute_loss(v_x, v_true_y)

                # compute evaluation scores
                v_pred_y = (v_pred_y > 0.).long()
                v_pred_y, v_true_y = v_pred_y.detach().cpu().numpy(), v_true_y.detach().cpu().numpy()
                accuracy = skm.accuracy_score(v_true_y, v_pred_y)
                balanced_accuracy = skm.balanced_accuracy_score(v_true_y, v_pred_y)
                zero_one_loss = skm.zero_one_loss(v_true_y, v_pred_y)
                hamming_loss = skm.hamming_loss(v_true_y, v_pred_y)
                precision = skm.precision_score(v_true_y, v_pred_y, average='samples')
                recall = skm.recall_score(v_true_y, v_pred_y, average='samples')
                f1 = skm.f1_score(v_true_y, v_pred_y, average='samples')
                f2 = skm.fbeta_score(v_true_y, v_pred_y, beta=2., average='samples')
                v_metrics.update({'accuracy': accuracy,
                                  'balanced_accuracy': balanced_accuracy,
                                  'zero_one_loss': zero_one_loss,
                                  'hamming_loss': hamming_loss,
                                  'precision': precision,
                                  'recall': recall,
                                  'f1': f1,
                                  'f2': f2})

                self.send_to_dash(v_metrics, mode='val')
                del v_metrics

                self.model.train()

            # 3|3>>>> wrap up

            if DEBUG:
                last_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"lr ={last_lr} after {self.iters_so_far} gradient steps")

            self.iters_so_far += 1

        self.epochs_so_far += 1

    def test(self, dataloader):

        self.model.eval()

        for i, (x, true_y) in enumerate(tqdm(dataloader)):

            x, true_y = x.to(self.device), true_y.to(self.device)
            metrics, _, pred_y = self.compute_loss(x, true_y)

            # compute evaluation scores
            pred_y = (pred_y > 0.).long()
            pred_y, true_y = pred_y.detach().cpu().numpy(), true_y.detach().cpu().numpy()
            accuracy = skm.accuracy_score(true_y, pred_y)
            balanced_accuracy = skm.balanced_accuracy_score(true_y, pred_y)
            zero_one_loss = skm.zero_one_loss(true_y, pred_y)
            hamming_loss = skm.hamming_loss(true_y, pred_y)
            precision = skm.precision_score(true_y, pred_y, average='samples')
            recall = skm.recall_score(true_y, pred_y, average='samples')
            f1 = skm.f1_score(true_y, pred_y, average='samples')
            f2 = skm.fbeta_score(true_y, pred_y, beta=2., average='samples')
            metrics.update({'accuracy': accuracy,
                            'balanced_accuracy': balanced_accuracy,
                            'zero_one_loss': zero_one_loss,
                            'hamming_loss': hamming_loss,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'f2': f2})

            self.send_to_dash(metrics, mode='test')
            del metrics

        self.model.train()

    def save(self, path, epochs_so_far):
        model_dest_path = osp.join(path, f"model_{epochs_so_far}.tar")
        torch.save(self.model.state_dict(), model_dest_path)

    def load(self, path, epochs_so_far):
        model_orig_path = osp.join(path, f"model_{epochs_so_far}.tar")
        self.model.load_state_dict(torch.load(model_orig_path))
