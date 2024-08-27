# Import modules
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.distributions import Categorical


class Integrator(nn.Module):
    def __init__(self):
        super(Integrator, self).__init__()

    def forward(self,
                raw: torch.Tensor,
                zvals: torch.Tensor,
                rays_d: torch.Tensor):
        '''
        It computes the accumulated transmittance along each ray, in order to
        render pixels of the current frame.
        '''

        # Define a lambda function to compute alphas from samples
        raw2alpha = lambda raw, dists, act_fn = relu: 1 - torch.exp(-act_fn(raw) * dists)

        # Compute distances and reshape as [rays, samples]
        dists = zvals[..., 1:] - zvals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], dim=-1)
        dists *= torch.norm(rays_d[..., None,:], dim=-1)

        # Extracts channes values 
        chs = torch.sigmoid(raw[..., :-1])

        # Compute alphas and cumulative product
        alpha = raw2alpha(raw[..., -1], dists)
        tmp = torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], dim=-1)
        cumprod = torch.cumprod(tmp, dim=-1)[:, :-1]

        # Compute integration weights and channels values, as [rays, chs]
        weights = alpha * cumprod
        chs_map = torch.sum(weights[..., None] * chs, dim=-2)

        # Compute integration of weights and densities for depth, as [rays, depth]
        depth_map = torch.sum(weights * zvals, dim=-1) / torch.sum(weights, dim=-1)

        # Finally, compute weights sparsity loss
        # TODO: check if try-expect makes sense
        try:
            sparsity_loss = Categorical(
                probs=torch.cat([weights, 1.0-weights.sum(-1, keepdim=True)+1e-6], dim=-1)
            ).entropy()
        except:
            sparsity_loss = 0
            print("Warning: Sparsity Loss cannot be computed.")

        return chs_map, depth_map, sparsity_loss
