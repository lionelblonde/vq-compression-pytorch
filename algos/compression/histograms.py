import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_cross_entropy(input, target):
    """Cross-entropy with soft targets, e.g. result of targets=softmax(h).
    A priori not available as a bundled function in PyTorch at the time of writing.

    Both input and target are of size [BxHxW, D].
    """
    logq = -F.log_softmax(input, dim=-1)
    sce = (target * logq).sum(dim=-1)
    return sce.mean()


class Histogram(nn.Module):
    """Set of learnable parameters theta that can be interpreted as a histogram.
    These are updated by being fit to the indices of the encodings of the learned quantizer's codebook.
    The gradient-based update of the thetas happens through the cross-entropy loss only, while we take
    care of detaching the targets they are fit against. The purpose of the thetas is to give us
    information about the learned codebook's usage per vector, while the perplexity is an aggregated metric.

    How this works:
    + the thetas (aimed at approximating the encoding indices, i.e. codes) are learned with the cross-entropy loss
    + and are used as detached targets in the soft-entropy loss, which gives grads to the encoder and quantizer!
    """

    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        # Create set of learnable parameters thetas. They represent the probabilities of each vector of the
        # learned quantizer's codebook being used. The thetas vector starts as the uniform probability vector
        self.thetas = nn.Parameter(torch.ones(self.hps.c_num) / self.hps.c_num)  # has size [c_num]

    def ce(self, targets):
        """Cross-entropy loss"""
        # The targets used are the encoding indices (index of vector in learned codebook)

        # Flatten the encoding indices, from size [BxHxW, 1] to [BxHxW] (becaused we unqueezed it before!)
        targets = targets.flatten()

        # Detach the encoding indices (just in case, even if the argmin is already non-differentiable)
        # to prevent this loss from updating the learned codebook in the vector quantizer
        targets.detach()

        # Expand the thetas from the size [D] to [BxHxW, D] (D is c_num)
        params = self.thetas.expand(targets.shape[0], -1)  # if non-singleton dim, expand unqueezes at 0 by default

        # Return the cross-entropy loss between the learned parameters thetas and the detached encoding indices
        return F.cross_entropy(params, targets)

    def soft_ce(self, targets):
        """Soft version of the cross-entropy loss"""
        # The targets used are the normalized proximities between the learned codebook vectors and the latents

        # NOTE: while the cross-entropy loss above was updated only the thetas parameters
        # this soft loss instead detaches the thetas and urges the distribution of proximities between
        # latent the codebook vectors and latent to be closer to the per-vector codebook usage learned by the thetas

        # Why send such gradients to the quantizer? Because it gives more signal to each vector of the codebook
        # about how to change for its distribution of proximities to match the distribution of actual usage

        # The intended effect is that the latents and codes will learn to concentrate together
        # effectively making the entropy of the learned codebook smaller (like w/ perplexity, lower is better)

        # Convert targets from BCHW to BHWC
        targets = targets.movedim(1, -1).contiguous()  # or equivalently: targets.permute(0, 2, 3, 1).contiguous()
        # Flatten the tensor to be 2-dimensional, with size [BxHxW, D]  (D is c_num)
        targets = targets.flatten(start_dim=0, end_dim=-2)

        # Set the thetas to be non-learned parameters
        non_learned_params = self.thetas.detach()

        # Expand the thetas from the size [D] to [BxHxW, D] (as we have seen above is the size of targets)
        non_learned_params = non_learned_params.expand_as(targets)

        # Return the "soft" version of the cross-entropy loss
        return soft_cross_entropy(non_learned_params, targets)
