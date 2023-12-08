import torch
import torch.nn as nn
import torch.nn.functional as func


class VectorQuantizer(nn.Module):

    def __init__(self, hps, device):
        super().__init__()
        self.hps = hps
        self.device = device
        self.use_vqvae_internals = False  # we don't care much about the original vqvae internals > posterity
        self.commitment_cost = 0.25  # only used if using the vq loss from the vqvae original paper
        # Create the embedding, with number and embeddings and embedding dimensions as parameters
        self.centers = nn.Embedding(hps.c_num, hps.z_channels)  # dimension is [D, C]
        torch.nn.init.uniform_(self.centers.weight, a=hps.c_min, b=hps.c_max)

    def forward(self, z):
        # Convert latents z from BCHW to BHWC
        z = z.movedim(1, -1).contiguous()  # or equivalently: z.permute(0, 2, 3, 1).contiguous()
        # Save this shape for later
        latent_shape = z.shape

        # Create a clone to use later
        z_orig = z.clone()

        # Flatten the tensor to be 2-dimensional, with size [BxHxW, C] (C is z_channels)
        z = z.flatten(start_dim=0, end_dim=2)
        # Inflate along the last dimension, by a value equal to thenumber of embeddings c_num
        # In effect, we first have a [BxHxW, C, 1] tensor, then a [BxHxW, C, D]
        z = z[:, :, None].expand(-1, -1, self.hps.c_num)  # could also use unsqueeze for the first bit
        # We transpose the centers embedding from [D, C] to [C, D], unsqueeze the first dim => [1, C, D]
        # and expand as z which is now [BxHxW, C, D]. The resulting reformed centers are then [BxHxW, C, D] too
        centers = self.centers.weight.T[None, :, :].expand_as(z)
        # Compute the distances between the latents z and the centers embeddings
        distances = torch.linalg.vector_norm(z - centers, dim=1)  # this is of size [BxHxW, D]

        # Get the index of the nearest codebook vector =center,
        # i.e. the vector of the embedding closest to the latent z; this has size [BxHxW]
        encoding_indices = torch.argmin(distances, dim=-1)  # could also use dim=1 since 2-dim

        # Create also a "soft" version of the encoding indices: normalized distances w.r.t. codebook centers
        normalized_proximities = func.softmax(-distances, dim=-1)  # this is of size [BxHxW, D]

        # Quantize the latents z
        qz = self.centers(encoding_indices)
        # Unflatten the quantized latents into the original latent shape
        qz = qz.view(latent_shape)

        # Create the encodings from the obtained encoding indices (closest vectors from embedding)
        encoding_indices = encoding_indices.unsqueeze(dim=1)  # this has size [BxHxW, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self.hps.c_num).to(self.device)   # this has size [BxHxW, D]
        encodings.scatter_(1, encoding_indices, 1)  # scatter is the reverse operation of gather
        # the operation does not change the size; encodings still has size [BxHxW, D]

        if self.use_vqvae_internals:

            # Compute the vector-quantization losses
            q_loss = func.mse_loss(z_orig.detach(), qz)  # codebook loss
            e_loss = func.mse_loss(z_orig, qz.detach())  # commitment loss
            vq_loss = q_loss + (self.commitment_cost * e_loss)

            # Make the quantized latents equal to themselves during the forward pass,
            # and to the latents during the backward pass. In effect, this makes
            # the gradient w.r.t. latents be equal to the gradient w.r.t. quantized latents.
            # This is akin to a straight-through estimation, and it is used in the vqvae paper
            qz = z_orig + (qz - z_orig).detach()

        else:

            # No extra loss introduced here to compensate for the lack of gradients
            vq_loss = 0.

            # Make the quantized latents equal to themselves during the forward pass,
            # and to the a soft approximation of them during the backward pass.
            soft_qz = (normalized_proximities[:, None, :] * centers).sum(dim=-1)  # broadcasting happening
            # at the None from 1 to C because first term is [BxHxW, 1, D] while centers has size [BxHxW, D]
            # Unflatten and convert the soft quantized latents soft_qz from BHWC to BCHW
            soft_qz = soft_qz.view(latent_shape)
            qz = soft_qz + (qz - soft_qz).detach()

        # Convert the quantized latents qz from BHWC to BCHW
        qz = qz.movedim(-1, 1).contiguous()
        # Do the same for the normalized distances
        normalized_proximities = normalized_proximities.movedim(-1, 1).contiguous()

        # Compute the average probability for each vector of the learned codebook of being picked
        avg_probs = encodings.mean(dim=0)  # this is of size [D]
        # Compute the perplexity
        # from the Wiki page: the perplexity is a measurement of how well a probability distribution
        # predicts a sample. It may be used to compare probability models. A low perplexity indicates
        # the probability distribution is good at predicting the sample.
        perplexity = (-avg_probs * (avg_probs + 1e-10).log()).sum().exp()  # gives an idea of codebook usage

        return qz, vq_loss, perplexity, encoding_indices, normalized_proximities
