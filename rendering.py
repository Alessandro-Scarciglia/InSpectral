# Import modules
import torch
import torch.nn as nn

# Import custom modules
from sampler import Sampler
from embedder import HashEmbedder
from sh_encoder import SHEncoder
from small_nerf import NeRFSmall
from integrator import Integrator

# Parameters
from parameters import *
from rays_generator import RaysGenerator
import matplotlib.pyplot as plt
import numpy as np
import json


class NeuralRenderer(nn.Module):
    def __init__(self,
                 n_ray_samples: int, near: float, far: float,
                 bbox: tuple, n_levels: int, n_features_per_level: int, log2_hashmap_size: int, low_res: int, high_res: int,
                 input_dim: int, degree: int, out_dim: int,
                 n_layers: int, hidden_dim: int, geo_feat_dim: int, n_layers_color: int, hidden_dim_color: int, input_ch: int, input_ch_views: int, out_ch: int,
                 device: str = 'cpu'):
        super(NeuralRenderer, self).__init__()

        # Attributes
        self.device = torch.device(device)
        
        # Generate a set of samples uniformly distributed along the ray,
        # within the [near, far] interval.
        self.sampler = Sampler(n_samples=n_ray_samples,
                               near=near,
                               far=far,
                               device=device)
        
        # Generate embeddings of an input coordinate
        self.embedder = HashEmbedder(bbox=bbox,
                                     n_levels=n_levels,
                                     n_features_per_level=n_features_per_level,
                                     log2_hashmap_size=log2_hashmap_size,
                                     low_resolution=low_res,
                                     high_resolution=high_res,
                                     device=device)
        
        # Generate encoding for view directions
        self.sh_encoder = SHEncoder(input_dim=input_dim,
                                    degree=degree,
                                    out_dim=out_dim,
                                    device=device)
        
        # Infer the density and channel values for each sample along a ray
        self.nerf = NeRFSmall(n_layers=n_layers,
                              hidden_dim=hidden_dim,
                              geo_feat_dim=geo_feat_dim,
                              n_layers_color=n_layers_color,
                              hidden_dim_color=hidden_dim_color,
                              input_ch=input_ch,
                              input_ch_views=input_ch_views,
                              out_ch=out_ch,
                              device=device).to(device)
        
        # Integrates all densities and channels values along a ray to get the 
        # overall properties of a single pixel.
        self.integrator = Integrator(device=device)


    def forward(self, rays: torch.Tensor):

        # Bring rays to target device
        rays = rays.to(self.device)

        # Preprocessing steps are carried out without retaining the gradient
        with torch.no_grad():

            # Produce samples along each ray produced from camera
            samples, zvals = self.sampler(rays[..., :3], rays[..., 3:6])
            samples = samples.to(self.device)
            zvals = zvals.to(self.device)

            # Concatenate points with view direction and kill one dimension
            viewdirs = rays[..., 3:6].unsqueeze(1).to(self.device)
            viewdirs = viewdirs.expand(-1, samples.shape[1], -1)
            rays_and_viewdirs = torch.cat([samples, viewdirs], dim=-1)
            rays_and_viewdirs = rays_and_viewdirs.reshape(-1, 6)

        # Hash-encode the samples coordinates and SH-encode the view directions
        enc_points, keep_mask = self.embedder(rays_and_viewdirs[..., :3])
        enc_dirs = self.sh_encoder(rays_and_viewdirs[..., 3:6])
        
        # Concatenate as a whole input vector and compute a forward pass with SmallNeRF
        input_vector = torch.cat([enc_points, enc_dirs], dim=-1).to(self.device)
        output = self.nerf(input_vector)

        # Clean (i.e. set sigma to 0) output estimates out of bbox boundaries
        # and reshape as [points, samples, channels]
        output = output.to(self.device)
        output[~keep_mask, -1] = 0
        output = output.reshape(rays.shape[0], self.sampler.n_samples, -1)
        
        # Integrate densities and channels values estimate along each ray
        chs_map, depth_map, sparsity_loss = self.integrator(output, zvals, rays[..., 3:6])

        return chs_map, depth_map, sparsity_loss
    


if __name__ == "__main__":

    # Retrieve intrinsic calibration matrix K
    with open('calibration/calibration.json', "r") as fopen:
        calib = np.array(json.load(fopen)["mtx"]).reshape(3, 3)
        calib[:2, :2] /=  16.

    raygen = RaysGenerator(**rays_parameters, K=calib)
    model = NeuralRenderer(**sampler_parameters,
                            **sh_parameters,
                            **hash_parameters,
                            **nerf_parameters,
                            device="cuda:0")
    
    c2w1 = torch.tensor([[
                6.123233995736766e-17,
                0.0,
                -1.0,
                2.0
            ],
            [
                -0.8660254037844387,
                -0.4999999999999998,
                -5.3028761936245346e-17,
                0.0
            ],
            [
                -0.4999999999999998,
                0.8660254037844387,
                -3.061616997868382e-17,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]], dtype=torch.float32)
    
    c2w2 = torch.tensor([[
                6.123233995736766e-17,
                0.0,
                -1.0,
                2.0
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0
            ],
            [
                1.0,
                0.0,
                6.123233995736766e-17,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]], dtype=torch.float32)
    
    rays = raygen(c2w2).reshape(-1, 6)
    
    rgb, depth, _ = model(rays)
