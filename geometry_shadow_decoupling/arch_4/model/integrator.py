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
            near_depth: float = 0.00,
            far_depth: float = 1.125,
            device: str = 'cpu'
    ):
        super(Integrator, self).__init__()

        # Attributes
        self.near_depth = near_depth
        self.far_depth = far_depth
        self.device = torch.device(device)

    def forward(
            self,
            raw: torch.Tensor,
            zvals: torch.Tensor,
            rays_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the accumulated transmittance along each ray to output integrated 
        channels (e.g. RGB), depth maps, a structural sparsity loss, and a 
        mathematically valid 0-1 segmentation opacity mask.
        """
        # Step 1: Cast all volumetric tracking inputs safely onto the active execution device
        raw = raw.to(self.device)
        zvals = zvals.to(self.device)
        rays_d = rays_d.to(self.device)

        # Step 2: Define vectorized alpha calculation via local continuous density activation
        raw2alpha = lambda raw_val, distance_val: 1. - torch.exp(-F.relu(raw_val) * distance_val)

        # Step 3: Compute localized coordinate spatial intervals (delta steps) along the rays
        dists = zvals[..., 1:] - zvals[..., :-1]
        dists = torch.cat([dists, torch.full(dists[..., :1].shape, 1e10, device=self.device)], dim=-1)
        dists *= torch.norm(rays_d[..., None, :], dim=-1)

        # Step 4: Structurally decouple the raw feature grid into separate color and density tensors
        num_color_channels = cfg_parameters["channels"]
        chs = raw[..., :num_color_channels]
        density = raw[..., num_color_channels]

        # Step 5: Compute continuous opacity bounds and aggregate cumulative transmittance vectors
        alpha = raw2alpha(density, dists)
        tmp = torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], dim=-1)
        cumprod = torch.cumprod(tmp, dim=-1)[:, :-1]

        # Step 6: Extract matching sampling integration weights across the ray vectors
        weights = alpha * cumprod
        
        # Step 7: Mathematically accumulate structural weights to extract the core segmentation mask
        accumulated_opacity = torch.sum(weights, dim=-1)
        mask = torch.clamp(accumulated_opacity, min=0.0, max=1.0).unsqueeze(-1)
        
        # Step 8: Linearly blend isolated color vectors into localized pixel color maps
        chs_map = torch.sum(weights[..., None] * chs, dim=-2)
        chs_map = torch.clamp(input=chs_map, min=0.0, max=1.0)

        # Step 9: Reconstruct normalized geometric distance depth fields using accumulated opacity
        # Compute volumetric depth (no artificial normalization)
        depth_map = torch.sum(weights * zvals, dim=-1)

        # Normalize to [0,1] range based on near/far bounds
        depth_map = (depth_map - self.near_depth) / (self.far_depth - self.near_depth)

        # Invert mapping: near -> white (1), far -> black (0)
        depth_map = 1.0 - depth_map

        # Suppress background using accumulated opacity (prevents white background artifacts)
        depth_map = depth_map * accumulated_opacity

        # Final clamp for image encoding
        depth_map = torch.clamp(depth_map, 0.0, 1.0)

        # Step 10: Enforce true Shannon Entropy Sparsity over ray distribution profiles
        # This penalizes multi-peak or hazy weight assignments, clearing out foggy artifacts.
        entropy = -1.0 * weights * torch.log(weights + 1e-10)
        sparsity_loss = torch.mean(torch.sum(entropy, dim=-1))

        return chs_map, depth_map, sparsity_loss, mask