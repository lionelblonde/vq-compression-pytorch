from collections import defaultdict

import torch.nn as nn

from algos.compression.models import EncoderModel, DecoderModel
from algos.compression.quantizers import VectorQuantizer
from algos.compression.histograms import Histogram


class VectorQuantizationAutoEncoder(nn.Module):

    def __init__(self, hps, device):
        super().__init__()
        self.hps = hps
        self.device = device
        self.encoder = EncoderModel(self.hps).to(self.device)
        self.decoder = DecoderModel(self.hps).to(self.device)
        self.vector_quantizer = VectorQuantizer(self.hps, self.device).to(self.device)
        self.histogram = Histogram(self.hps).to(self.device)

    def forward(self, x):
        z = self.encoder(x)
        qz, vq_loss, perplexity, encoding_indices, normalized_proximities = self.vector_quantizer(z)
        recon_x = self.decoder(qz)
        return recon_x, vq_loss, perplexity, encoding_indices, normalized_proximities

    def loss_func(self, x, recon_x, vq_loss, encoding_indices, normalized_proximities):

        recon_error = nn.functional.mse_loss(x, recon_x)

        losses = defaultdict()
        losses['recon_error'] = recon_error
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
