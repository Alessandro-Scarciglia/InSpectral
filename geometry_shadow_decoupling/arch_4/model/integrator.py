# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_parameters import *


class Integrator(nn.Module):
    """
    This class implements the integration of all estimates made along the ray directions 
    and computes the transmittance integral to retrieve the final pixel properties.
    """
    def __init__(
            self,
            near: float,
            far: float,
            device: str = 'cpu'
    ):
        super(Integrator, self).__init__()

        # Attributes
        self.near = near
        self.far = far
        self.device = torch.device(device)

    def forward(
            self,
            raw: torch.Tensor,
            zvals: torch.Tensor,
            rays_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the accumulated transmittance along each ray.
        """
        # Bring inputs to target device immediately
        raw = raw.to(self.device)
        zvals = zvals.to(self.device)
        rays_d = rays_d.to(self.device)

        # Lambda function to compute alphas from raw densities (applies ReLU here!)
        raw2alpha = lambda raw_val, distance_val: 1. - torch.exp(-F.relu(raw_val) * distance_val)

        # Compute distances between samples along the rays
        dists = zvals[..., 1:] - zvals[..., :-1]
        dists = torch.cat([dists, torch.full(dists[..., :1].shape, 1e10, device=self.device)], dim=-1)
        dists *= torch.norm(rays_d[..., None, :], dim=-1)

        # Extract predicted color channels
        chs = raw[..., :cfg_parameters["channels"]]

        # Compute alphas and transmittance cumulative product
        alpha = raw2alpha(raw[..., -1], dists)
        tmp = torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], dim=-1)
        cumprod = torch.cumprod(tmp, dim=-1)[:, :-1]

        # Compute ray integration weights
        weights = alpha * cumprod
        
        # Build the segmentation mask from accumulated weights
        mask = torch.clamp(torch.sum(weights, dim=-1), min=0.0, max=1.1).unsqueeze(-1)
        
        # Compute final RGB pixel color map
        chs_map = torch.sum(weights[..., None] * chs, dim=-2)
        chs_map = torch.clamp(chs_map, 0.0, 1.0)

        # Compute depth map integrated along the ray
        zvals_norm = (self.far - zvals) / (self.far - self.near)
        depth_map = torch.sum(weights * zvals_norm, dim=-1) / (torch.sum(weights, dim=-1) + 1e-5)
        depth_map = torch.clamp(depth_map, 0.0, 1.0)

        # --- SPARSITY LOSS L1 ---
        # Spinge i pesi di campionamento a zero lungo le zone vuote del raggio,
        # impedendo la formazione di nebbia senza mandare in crash l'ottimizzazione iniziale.
        sparsity_loss = torch.mean(torch.abs(weights))

        return chs_map, depth_map, sparsity_loss, mask