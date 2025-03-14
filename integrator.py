# Import modules
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.distributions import Categorical
from parameters_synth import *


class Integrator(nn.Module):
    def __init__(self,
                 device : str = 'cpu'):
        super(Integrator, self).__init__()

        # Attributes
        self.device = torch.device(device)

    def forward(self,
                raw: torch.Tensor,
                zvals: torch.Tensor,
                rays_d: torch.Tensor):
        '''
        It computes the accumulated transmittance along each ray, in order to
        render pixels of the current frame.
        '''

        # Bring inputs to target device
        raw = raw.to(self.device)
        zvals = zvals.to(self.device)
        rays_d = rays_d.to(self.device)

        # Define a lambda function to compute alphas from samples
        raw2alpha = lambda raw, dists, act_fn = relu: 1. - torch.exp(-act_fn(raw) * dists)

        # Compute distances and reshape as [rays, samples]
        dists = zvals[..., 1:] - zvals[..., :-1]
        dists = torch.cat([dists, torch.full(dists[..., :1].shape, 1e10, device=self.device)], dim=-1)
        dists *= torch.norm(rays_d[..., None, :], dim=-1)

        # Extracts channes values 
        chs = raw[..., :-1]#torch.sigmoid(raw[..., :-1])

        # Compute alphas and cumulative product
        alpha = raw2alpha(raw[..., -1], dists)
        tmp = torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], dim=-1)
        cumprod = torch.cumprod(tmp, dim=-1)[:, :-1]

        # Compute integration weights and channels values, as [rays, chs]
        weights = alpha * cumprod
        chs_map = torch.sum(weights[..., None] * chs, dim=-2) 

        # Compute integration of weights and densities for depth, as [rays, depth]
        zvals = (sampler_parameters["far"] - zvals) / (sampler_parameters["far"] - sampler_parameters["near"])
        depth_map = torch.sum(weights * zvals, dim=-1) / (torch.sum(weights, dim=-1) + 1e-5)

        # Finally, compute weights sparsity loss
        try:
            sparsity_loss = Categorical(
                probs = torch.cat([weights, 1.0 - weights.sum(-1, keepdim=True) + 1e-6], dim=-1)
            ).entropy().sum()
        except:
            sparsity_loss = torch.tensor(0.0, device=self.device)
            print("Warning: Sparsity Loss cannot be computed.")


        return chs_map, depth_map, sparsity_loss
