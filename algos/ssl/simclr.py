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
from helpers.metrics_util import compute_classif_eval_metrics
from helpers.model_util import add_weight_decay
from algos.ssl.models import SimCLRModel
from algos.ssl.ntx_ent_loss import NTXentLoss
from algos.ssl.lars import LARSWrapper


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


class SimCLR(object):

    def __init__(self, device, hps):
        self.device = device
        self.hps = hps

        self.iters_so_far = 0
        self.epochs_so_far = 0

        if self.hps.clip_norm <= 0:
            logger.info(f"clip_norm={self.hps.clip_norm} <= 0, hence disabled.")

        self.model = SimCLRModel(
            backbone_name=self.hps.backbone,
            backbone_pretrained=self.hps.pretrained_w_imagenet,
            fc_hid_dim=self.hps.fc_hid_dim,
            fc_out_dim=self.hps.fc_out_dim,
        ).to(self.device)

        self.criterion = NTXentLoss(temperature=self.hps.ntx_temp).to(self.device)
        self.sim = nn.CosineSimilarity(dim=2).to(self.device)
        self.bce = nn.BCEWithLogitsLoss().to(self.device)

        layerwise_lr_adaption = True  # can do better
        if not layerwise_lr_adaption:
            self.opt = torch.optim.Adam(
                add_weight_decay(
                    self.model,
                    weight_decay=self.hps.wd,
                ),
                lr=self.hps.lr,
                weight_decay=0.,  # added on per-param basis in groups manually
            )  # upgrade: "decay the learning rate with the cosine decay schedule without restarts"
        else:
            self.opt = torch.optim.SGD(
                add_weight_decay(
                    self.model,
                    weight_decay=self.hps.wd,
                ),
                lr=2.4,
                momentum=0.9,
                nesterov=False,
                weight_decay=0.,  # added on per-param basis in groups manually
            )  # upgrade: "decay the learning rate with the cosine decay schedule without restarts"
            self.opt = LARSWrapper(
                opt=self.opt,
                trust_coeff=1e-3,
            )  # wrap with the LARC-inspired LARS wrapper
            logger.info("using LARS optimizer wrapper")

        self.ctx = (
            torch.amp.autocast(device_type='cuda', dtype=torch.float16 if self.hps.fp16 else torch.float32)
            if self.hps.cuda
            else nullcontext()
        )

        self.scaler = gs.GradScaler(enabled=self.hps.fp16)

        if self.hps.load_checkpoint is not None:
            # load model weights from a checkpoint file
            self.load_from_path(self.hps.load_checkpoint)
            logger.info("model loaded")

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

    def train(self, train_dataloader, val_dataloader, knn_dataloader):

        agg_iterable = zip(
            tqdm(train_dataloader),
            itertools.chain.from_iterable(itertools.repeat(val_dataloader)),
            strict=False,
        )

        for i, ((t_x, _), (v_x, v_true_y)) in enumerate(agg_iterable):

            t_x_i, t_x_j = [e.squeeze() for e in torch.tensor_split(t_x, 2, dim=1)]  # unpack

            if self.hps.cuda:
                t_x_i = t_x_i.pin_memory().to(self.device, non_blocking=True)
                t_x_j = t_x_j.pin_memory().to(self.device, non_blocking=True)
            else:
                t_x_i, t_x_j = t_x_i.to(self.device), t_x_j.to(self.device)

            with self.ctx:
                t_metrics, t_loss = self.compute_loss(t_x_i, t_x_j)
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

                    z_bank = []
                    y_bank = []

                    logger.info("building bank")
                    for j, (k_x, k_true_y) in enumerate(knn_dataloader):
                        if j >= 10:
                            break  # takes too long otherwise; shuffle is on
                        if self.hps.cuda:
                            k_x = k_x.pin_memory().to(self.device, non_blocking=True)
                        else:
                            k_x = k_x.to(self.device)
                        with self.ctx:
                            k_z = self.model.mono_forward(k_x)
                        z_bank.append(k_z)
                        y_bank.append(k_true_y)
                    logger.info("bank built")
                    num_classes = k_true_y.size(-1)  # using the last label loaded
                    z_bank = torch.cat(z_bank, dim=0).to(self.device)  # size: (NxD)
                    y_bank = torch.cat(y_bank, dim=0).to(self.device)  # size: (NxC)

                    if self.hps.cuda:
                        v_x = v_x.pin_memory().to(self.device, non_blocking=True)
                        v_true_y = v_true_y.pin_memory().to(self.device, non_blocking=True)
                    else:
                        v_x, v_true_y = v_x.to(self.device), v_true_y.to(self.device)

                    with self.ctx:

                        v_z = self.model.mono_forward(v_x)  # size: (BxD)

                        sim = self.sim(v_z.unsqueeze(1), z_bank.unsqueeze(0)) / 0.25  # temperature
                        # sizes: (BxD), (NxD) -> (BxN) with the sim reduction along dim 2

                        sim_weights, sim_indices = sim.topk(k=50, dim=-1)  # only keep k closest images
                        # size: (BxK) by keeping only a subset of the N's

                        # look for the labels of the k closest images
                        sim_y = torch.gather(
                            y_bank.unsqueeze(0).repeat(self.hps.batch_size, 1, 1),
                            dim=1,
                            index=sim_indices.unsqueeze(-1).repeat(1, 1, num_classes),
                        )
                        # size: (BxKxC) by selecting the K indices within dim 1 of the expansion (BxNxC)

                        sim_weighted_y = (sim_y * sim_weights.exp().unsqueeze(-1)).sum(dim=1)

                        v_loss = self.bce(sim_weighted_y, v_true_y)

                        v_metrics = {'loss': v_loss.item()}

                    self.send_to_dash(v_metrics, mode='val')
                    del v_metrics

                self.model.train()

            self.iters_so_far += 1

        self.epochs_so_far += 1

    def test(self, test_dataloader, knn_dataloader):
        self.model.eval()

        with torch.no_grad():

            z_bank = []
            y_bank = []

            logger.info("building bank")  # compared to val, only need to build once
            for j, (k_x, k_true_y) in enumerate(knn_dataloader):
                if j >= 10:
                    break  # takes too long otherwise; shuffle is on
                if self.hps.cuda:
                    k_x = k_x.pin_memory().to(self.device, non_blocking=True)
                else:
                    k_x = k_x.to(self.device)
                with self.ctx:
                    k_z = self.model.mono_forward(k_x)
                z_bank.append(k_z)
                y_bank.append(k_true_y)
            logger.info("bank built")
            num_classes = k_true_y.size(-1)  # using the last label loaded
            z_bank = torch.cat(z_bank, dim=0)  # size: (NxD)
            y_bank = torch.cat(y_bank, dim=0)  # size: (NxC)

            for x, true_y in tqdm(test_dataloader):

                if self.hps.cuda:
                    x = x.pin_memory().to(self.device, non_blocking=True)
                    true_y = true_y.pin_memory().to(self.device, non_blocking=True)
                else:
                    x, true_y = x.to(self.device), true_y.to(self.device)

                with self.ctx:

                    z = self.model.mono_forward(x)  # size: (BxD)

                    sim = self.sim(z.unsqueeze(1), z_bank.unsqueeze(0)) / 0.25  # temperature
                    # sizes: (BxD), (NxD) -> (BxN) with the sim reduction along dim 2

                    sim_weights, sim_indices = sim.topk(k=50, dim=-1)  # only keep k closest images
                    # size: (BxK) by keeping only a subset of the N's

                    # look for the labels of the k closest images
                    sim_y = torch.gather(
                        y_bank.unsqueeze(0).repeat(self.hps.batch_size, 1, 1),
                        dim=1,
                        index=sim_indices.unsqueeze(-1).repeat(1, 1, num_classes),
                    )
                    # size: (BxKxC) by selecting the K indices within dim 1 of the expansion (BxNxC)

                    sim_weighted_y = (sim_y * sim_weights.exp().unsqueeze(-1)).sum(dim=1)

                    loss = self.bce(sim_weighted_y, true_y)

                    metrics = {'loss': loss.item()}

                self.send_to_dash(metrics, mode='test')
                del metrics

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

        # Create new head
        self.new_head = nn.Linear(
            self.model.tv_backbone_inner_fc_dim,
            self.hps.num_classes,  # classification downstream task
            bias=True,
        ).to(self.device)

        # Set the model as the backbone itself
        self.model = self.model.backbone  # not to carry around the "backbone", prone to omission

        # Neutralize model before replacing the head
        for _, (_, p) in enumerate(self.model.named_parameters()):  # leave like this for quicker diagnostics
            p.requires_grad = False
        # this step is crucial despite optimizing only the new head; see discussion on PyTorch's official forum here:
        # https://discuss.pytorch.org/t/difference-between-set-parameter-requires-grad-false-and-exclude-them-from-optimizer/162126

        # Replace the entire mlp part of the SimCLR model with the created linear probe
        self.model.fc = self.new_head  # models are mutable like list and dict

        i = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                i += 1
                logger.info(f"param {n} is trainable!")
        assert i == 2, "too many trainable params (>2)"

        logger.info("logging the backbone after replacing the head")
        log_module_info(logger, 'simclr_model_with_new_head', self.model)

        # By this design, the resulting network has the exact same architecture as the classifier model!
        # they are therefore directly comparable!

        self.new_criterion = nn.BCEWithLogitsLoss().to(self.device)

        if self.hps.linear_probe:
            self.new_opt = torch.optim.SGD(
                add_weight_decay(self.new_head, weight_decay=2e-5),
                # this optimizer can only update the probe/new head!
                weight_decay=0.,  # added on per-param basis in groups manually
                lr=1.6,
                momentum=0.9,
                nesterov=True,
            )
        else:  # then `self.hps.fine_tuning` is True
            self.new_opt = torch.optim.SGD(
                add_weight_decay(self.model, weight_decay=2e-5),
                # this optimizer can update the entire model! => we use a lower lr
                weight_decay=0.,  # added on per-param basis in groups manually
                lr=0.8,
                momentum=0.9,
                nesterov=True,
            )

        self.new_ctx = (
            torch.amp.autocast(device_type='cuda', dtype=torch.float16 if self.hps.fp16 else torch.float32)
            if self.hps.cuda
            else nullcontext()
        )

        # Set up the gradient scaler for fp16 gpu precision
        self.new_scaler = gs.GradScaler(enabled=self.hps.fp16)

        # Reset the counters
        self.iters_so_far = 0
        self.epochs_so_far = 0

    def compute_classifier_loss(self, x, true_y):
        pred_y = self.model(x)
        loss = self.new_criterion(pred_y, true_y)
        metrics = {'loss': loss}
        return metrics, loss, pred_y

    def finetune_or_train_probe(self, train_dataloader, val_dataloader):
        # the code that follows is identical whether we fine-tune or just train the probe
        # because the only thing that changes between the two is the new optimizer (cf. above)

        special_key = 'finetune-probe'

        agg_iterable = zip(
            tqdm(train_dataloader),
            itertools.chain.from_iterable(itertools.repeat(val_dataloader)),
            strict=False,
        )

        for i, ((t_x, t_true_y), (v_x, v_true_y)) in enumerate(agg_iterable):

            if self.hps.cuda:
                t_x = t_x.pin_memory().to(self.device, non_blocking=True)
                t_true_y = t_true_y.pin_memory().to(self.device, non_blocking=True)
            else:
                t_x, t_true_y = t_x.to(self.device), t_true_y.to(self.device)

            with self.new_ctx:
                t_metrics, t_loss, _ = self.compute_classifier_loss(t_x, t_true_y)
                t_loss /= self.hps.acc_grad_steps

            self.new_scaler.scale(t_loss).backward()

            if ((i + 1) % self.hps.acc_grad_steps == 0) or (i + 1 == len(train_dataloader)):

                if self.hps.clip_norm > 0:
                    self.new_scaler.unscale_(self.new_opt)
                    cg.clip_grad_norm_(self.model.parameters(), self.hps.clip_norm)

                self.new_scaler.step(self.new_opt)
                self.new_scaler.update()
                self.new_opt.zero_grad()

                self.send_to_dash(t_metrics, mode=f"{special_key}-train")
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

                        v_metrics, _, v_pred_y = self.compute_classifier_loss(v_x, v_true_y)

                        # compute evaluation scores
                        v_pred_y = (v_pred_y >= 0.).long()
                        v_pred_y, v_true_y = v_pred_y.detach().cpu().numpy(), v_true_y.detach().cpu().numpy()
                        v_metrics.update(compute_classif_eval_metrics(v_true_y, v_pred_y))

                    self.send_to_dash(v_metrics, mode=f"{special_key}-val")
                    del v_metrics

                self.model.train()

            self.iters_so_far += 1

        self.epochs_so_far += 1

    def test_finetuned_or_probed_model(self, dataloader):
        # the code that follows is identical whether we fine-tune or just train the probe
        # because the only thing that changes between the two is the new optimizer (cf. above)

        special_key = 'finetune-probe'

        self.model.eval()

        with torch.no_grad():

            for x, true_y in tqdm(dataloader):

                if self.hps.cuda:
                    x = x.pin_memory().to(self.device, non_blocking=True)
                    true_y = true_y.pin_memory().to(self.device, non_blocking=True)
                else:
                    x, true_y = x.to(self.device), true_y.to(self.device)

                with self.ctx:

                    metrics, _, pred_y = self.compute_loss(x, true_y)

                    # compute evaluation scores
                    pred_y = (pred_y >= 0.).long()
                    pred_y, true_y = pred_y.detach().cpu().numpy(), true_y.detach().cpu().numpy()
                    metrics.update(compute_classif_eval_metrics(true_y, pred_y))

                self.send_to_dash(metrics, mode=f"{special_key}-test")
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

