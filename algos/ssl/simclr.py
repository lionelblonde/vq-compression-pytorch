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
from algos.ssl.models import SimCLRModel
from algos.ssl.ntx_ent_loss import NTXentLoss


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


class SimCLR(object):

    def __init__(self, device, hps, eval_every=100, new_head=False):
        self.device = device
        self.hps = hps

        self.iters_so_far = 0
        self.epochs_so_far = 0
        self.eval_every = eval_every

        if self.hps.clip_norm <= 0:
            logger.info(f"clip_norm={self.hps.clip_norm} <= 0, hence disabled.")

        assert int(self.hps.linear_probe) + int(self.hps.fine_tuning) == 1, "exactly 1 must be True"

        # Create nets
        self.model = SimCLRModel(
            backbone_name=self.hps.backbone,
            backbone_pretrained=self.hps.pretrained_w_imagenet,
            fc_hid_dim=self.hps.fc_hid_dim,
            fc_out_dim=self.hps.fc_out_dim,
        ).to(self.device)

        # Create criterion
        self.criterion = NTXentLoss(self.hps.batch_size).to(self.device)

        # Set up the optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hps.lr,
            weight_decay=self.hps.wd,
        )

        # Set up the gradient scaler for fp16 gpu precision
        self.scaler = gs.GradScaler(enabled=self.hps.fp16)

        # Set up lr scheduler. From original paper:
        # "decay the learning rate with the cosine decay schedule without restarts"
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.opt,
            T_max=self.hps.epochs,
            eta_min=0,
            last_epoch=-1
        )

        if new_head:
            self.renew_head()

        if hps.load_checkpoint is not None:
            # load model weights from a checkpoint file
            self.load_from_path(hps.load_checkpoint)

        log_module_info(logger, 'simclr_model', self.model)

    def compute_loss(self, x_i, x_j):
        z_i, z_j = self.model(x_i, x_j)  # positive pair
        loss = self.criterion(z_i, z_j)
        metrics = {'loss': loss.item()}
        return metrics, loss

    def send_to_dash(self, metrics, mode='unspecified'):
        wandb_dict = {f"{mode}/{k}": v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        wandb_dict['epoch'] = self.epochs_so_far
        wandb.log(wandb_dict, step=self.iters_so_far)

    def train(self, train_dataloader, val_dataloader):

        agg_iterable = zip(tqdm(train_dataloader), itertools.chain.from_iterable(itertools.repeat(val_dataloader)))

        for i, (((t_x_i, t_x_j), _), ((v_x_i, v_x_j), _)) in enumerate(agg_iterable):

            # 1|3>>>> train

            t_x_i, t_x_j = t_x_i.to(self.device), t_x_j.to(self.device)
            with am.autocast(enabled=self.hps.fp16):
                t_metrics, t_loss = self.compute_loss(t_x_i, t_x_j)

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

                v_x_i, v_x_j = v_x_i.to(self.device), v_x_j.to(self.device)
                v_metrics, _ = self.compute_loss(v_x_i, v_x_j)

                # 3|3>>>> wrap up
                self.send_to_dash(v_metrics, mode='val')
            self.send_to_dash(t_metrics, mode='train')
            
            if DEBUG:
                last_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"lr ={last_lr} after {self.iters_so_far} gradient steps")

            self.model.train()

            self.iters_so_far += 1

        self.epochs_so_far += 1

    def test(self, dataloader):
        self.model.eval()

        for i, ((x_i, x_j), _) in enumerate(tqdm(dataloader)):

            x_i, x_j = x_i.to(self.device), x_j.to(self.device)
            metrics, _ = self.compute_loss(x_i, x_j)

            self.send_to_dash(metrics, mode='test')

        # /!\ training mode is turned back on
        self.model.train()

    def renew_head(self):
        # In self-supervised learning, there are two ways to evaluate models:
        # (i) fine-tuning, and (ii) linear evaluation (or "linear probes").

        # In (i), the entire model is trained (backbone and other additional modules) without accessing the labels;
        # then a new linear layer is stacked on top of the backbone and both the backbone and the new linear layer
        # are trained by accessing the labels. Note, since the backbone has already been trained in the first stage,
        # it is common practice to use a smaller learning rate to avoid large shifts in weight space.
        # This is what the authors of the SimCLR paper are doing when they refer to "fine-tuning".

        # In (ii) a similar procedure is followed. In the first stage, models are trained without accessing the labels
        # (backbone and other additional modules); then in the second stage a new linear layer is stacked on top of the
        # backbone, and only this new linear layer is trained (no backprop on the backbone) by accessing the labels.

        self.new_head = nn.Linear(self.model.tv_backbone_inner_fc_dim, self.hps.num_classes).to(self.device)
        self.model.backbone.fc = self.new_head  # models are mutable like list and dict

        log_module_info(logger, 'simclr_model_with_new_head', self.model.backbone)

        # By this design, the resulting network has the exact same architecture as the classifier model!
        # they are therefore directly comparable!

        self.new_criterion = nn.CrossEntropyLoss().to(self.device)

        if self.hps.linear_probe:
            self.new_opt = torch.optim.Adam(
                self.new_head.parameters(),  # this optimizer can only update the probe/new head!
                lr=3e-4,
            )
        else:  # then `self.hps.fine_tuning` is True
            self.new_opt = torch.optim.Adam(
                self.model.parameters(),  # this optimizer can update the entire model! => we use a low lr
                lr=1e-5,
            )

        # Set up the gradient scaler for fp16 gpu precision
        self.new_scaler = gs.GradScaler(enabled=self.hps.fp16)

        # no need here for lr scheduler

        # Reset the counters
        self.iters_so_far = 0
        self.epochs_so_far = 0

    def compute_classifier_loss(self, x, true_y):
        pred_y = self.model.backbone(x)
        loss = self.new_criterion(pred_y, true_y)
        metrics = {'loss': loss.item()}
        return metrics, loss, pred_y

    def finetune_or_train_probe(self, train_dataloader, val_dataloader):
        # the code that follows is identical whether we fine-tune or just train the probe
        # because the only thing that changes between the two is the new optimizer (cf. above)

        special_key = 'finetune-probe'

        # from here onwards, semantically a priori identical to the classifier
        # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVvvVV

        samples = 0
        correct = 0

        agg_iterable = zip(tqdm(train_dataloader), itertools.chain.from_iterable(itertools.repeat(val_dataloader)))

        for i, ((t_x, t_true_y), (v_x, v_true_y)) in enumerate(agg_iterable):

            # 1|3>>>> train

            # t_x = t_x[0]  # needed if the number of transforms is set to >1

            t_x, t_true_y = t_x.to(self.device), t_true_y.to(self.device)
            with am.autocast(enabled=self.hps.fp16):
                t_metrics, t_loss, _ = self.compute_classifier_loss(t_x, t_true_y)

            # Update parameters
            self.new_opt.zero_grad()
            scaled_loss = self.new_scaler.scale(t_loss)
            scaled_loss.backward()
            if self.hps.clip_norm > 0:
                cg.clip_grad_norm_(self.model.parameters(), self.hps.clip_norm)
            self.new_scaler.step(self.new_opt)
            self.new_scaler.update()

            # 2|2>>>> evaluate

            self.model.eval()

            # v_x = v_x[0]  # needed if the number of transforms is set to >1

            v_x, v_true_y = v_x.to(self.device), v_true_y.to(self.device)
            logger.info(f"require={v_x.requires_grad}")
            v_metrics, _, v_pred_y = self.compute_classifier_loss(v_x, v_true_y)

            # accuracy

            samples_this_iter = v_true_y.size(0)
            samples += samples_this_iter

            quantized_v_pred_y = (v_pred_y > 0.5).float()
            correct_this_iter = (quantized_v_pred_y == v_true_y).sum().item()
            correct += correct_this_iter

            accuracy_this_iter = correct_this_iter / samples_this_iter
            v_metrics.update({'accuracy': accuracy_this_iter})

            # 3|3>>>> wrap up

            self.send_to_dash(t_metrics, mode=f"{special_key}-train")
            self.send_to_dash(v_metrics, mode=f"{special_key}-val")

            self.model.train()

            self.iters_so_far += 1

        self.epochs_so_far += 1

        accuracy_this_epoch = correct / samples
        logger.info(f"accuracy at the end of epoch {self.epochs_so_far} of finetuning/probing ={accuracy_this_epoch}")

    def test_finetuned_or_probed_model(self, dataloader):
        # the code that follows is identical whether we fine-tune or just train the probe
        # because the only thing that changes between the two is the new optimizer (cf. above)

        special_key = 'finetune-probe'

        # from here onwards, semantically a priori identical to the classifier
        # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVvvVV

        self.model.eval()

        samples = 0
        correct = 0

        for i, (x, true_y) in enumerate(tqdm(dataloader)):

            # x = x[0]  # needed if the number of transforms is set to >1

            x, true_y = x.to(self.device), true_y.to(self.device)
            metrics, _, pred_y = self.compute_classifier_loss(x, true_y)

            # accuracy

            samples_this_iter = true_y.size(0)
            samples += samples_this_iter

            quantized_pred_y = (pred_y > 0.5).float()
            correct_this_iter = (quantized_pred_y == true_y).sum().item()
            correct += correct_this_iter

            accuracy_this_iter = correct_this_iter / samples_this_iter
            metrics.update({'accuracy': accuracy_this_iter})

            self.send_to_dash(metrics, mode=f"{special_key}-test")

        # /!\ training mode is turned back on
        self.model.train()

    def save(self, path, epochs_so_far):
        model_dest_path = osp.join(path, f"model_{epochs_so_far}.tar")
        torch.save(self.model.state_dict(), model_dest_path)
        hps_dest_path = osp.join(path, f"hps_{epochs_so_far}.tar")
        torch.save(self.hps, hps_dest_path)

    def load(self, path, epochs_so_far):
        model_orig_path = osp.join(path, f"model_{epochs_so_far}.tar")
        self.model.load_state_dict(torch.load(model_orig_path))
    
    def load_from_path(self, path):
        self.model.load_state_dict(torch.load(path))
