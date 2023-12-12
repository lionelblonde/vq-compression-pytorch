from collections import defaultdict

import torch.nn as nn

from algos.compression.models import EncoderModel, DecoderModel
from algos.compression.quantizers import VectorQuantizer
from algos.compression.histograms import Histogram

from vector_quantize_pytorch import ResidualVQ  # from lucidrains


class VectorQuantizationAutoEncoder(nn.Module):

    def __init__(self, hps):
        super().__init__()
        self.hps = hps

        # XXX: this is temporary
        self.hps.residual_vq = True
        self.hps.num_quantizers = 8
        self.hps.quantize_dropout = False
        self.hps.codebook_size = 100
        self.hps.kmeans_iters = 100
        self.hps.threshold_ema_dead_code = 2
        self.hps.learnable_codebook = False

        self.encoder = EncoderModel(self.hps)
        self.decoder = DecoderModel(self.hps)

        if self.hps.residual_vq:
            # this follows algo 1 in the SoundStream paper
            self.vector_quantizer = ResidualVQ(
                # args specific to the residual extension
                num_quantizers=self.hps.num_quantizers,  # no defaults
                quantize_dropout=self.hps.quantize_dropout,  # default: False
                # args specific to the base vector quantizer(s)
                dim=self.hps.z_channels,  # no defaults
                codebook_size=self.hps.codebook_size,  # no defaults
                kmeans_init=True,  # default: False
                kmeans_iters=self.hps.kmeans_iters,  # default: 10
                threshold_ema_dead_code=self.hps.threshold_ema_dead_code,  # default: 0
                ema_update=True,  # default: True
                learnable_codebook=self.hps.learnable_codebook,  # default: False
            )
        else:
            # this is the implementation of a non-variational VQ-VAE
            self.vector_quantizer = VectorQuantizer(self.hps)

        self.histogram = Histogram(self.hps)

    def forward(self, x):
        z = self.encoder(x)

        if self.hps.residual_vq:
            z = z.permute(0, 2, 3, 1)
            sh = z.shape
            z = z.view(-1, sh[1] * sh[2], sh[3])
            qz, _, _ = self.vector_quantizer(z)
            qz = qz.permute(0, 2, 1)
            qz = qz.view(-1, sh[3], sh[1], sh[2])
            vq_loss = None
            perplexity = None
            encoding_indices = None
            normalized_proximities = None
        else:
            qz, vq_loss, perplexity, encoding_indices, normalized_proximities = self.vector_quantizer(z)

        recon_x = self.decoder(qz)
        return recon_x, vq_loss, perplexity, encoding_indices, normalized_proximities

    def loss_func(self, x, recon_x, vq_loss, encoding_indices, normalized_proximities):

        recon_error = nn.functional.mse_loss(x, recon_x)

        losses = defaultdict()
        losses['recon_error'] = recon_error
        loss = recon_error
        if not self.hps.residual_vq:
            losses['vq_loss'] = vq_loss
            loss = recon_error + vq_loss

            if self.hps.beta > 0:
                ce = self.histogram.ce(encoding_indices)
                loss += self.hps.beta * ce
                losses['ce'] = ce
            if self.hps.alpha > 0:
                soft_ce = self.histogram.soft_ce(normalized_proximities)
                loss += self.hps.alpha * soft_ce
                losses['soft_ce'] = soft_ce

        losses['loss'] = loss

        # Create a table containing the estimated codebook usage
        normalized_codex = nn.functional.softmax(self.histogram.thetas, dim=0)  # only one dim anyway
        indices_of_codex = list(range(len(normalized_codex)))
        table = [[c, p.item()] for c, p in zip(indices_of_codex, normalized_codex, strict=True)]

        return losses, table

