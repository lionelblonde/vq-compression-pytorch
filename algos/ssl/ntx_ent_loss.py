import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """NT-Xent, or Normalized Temperature-scaled Cross Entropy Loss,
    is the loss function introduced by Sohn, Kihyuk at NeurIPS 2016 in the paper
    "Improved Deep Metric Learning with Multi-class N-pair Loss Objective"
    https://proceedings.neurips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf
    Created as an alternative/improvement of the classical NCE loss of SimCLR
    """

    def __init__(self, normalize_hidden=True, temperature=0.07):  # default value use in MoCo
        super().__init__()
        self.normalize_hidden = normalize_hidden
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")  # we divide manually afterwards
        self.sim_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat
        the other 2(batch_size - 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_j.size(0)  # arbitrary choice of index
        device = z_j.device  # same

        z = torch.cat((z_i, z_j), dim=0)

        if self.normalize_hidden:
            z = nn.functional.normalize(z, p=2, dim=-1)

        sim = self.sim_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * batch_size, 1)

        mask = (torch.ones_like(sim, device=device) - torch.eye(2 * batch_size, device=device)).bool()

        negative_samples = sim.masked_select(mask).reshape(2 * batch_size, -1)

        loss = -(torch.log(positive_samples.exp() / negative_samples.exp().sum(dim=-1))).mean()

        return loss

