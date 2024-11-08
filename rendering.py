'''
This class encloses the design of the rendering model.
It takes as input the rays full info [origin, direction, cfg code] and returns
channels estimate, depth map and sparsity loss on sigmas distribution in the scene.
'''

# Import modules
import torch
import torch.nn as nn

# Import custom modules
from sampler import Sampler
from hash_embedder import HashEmbedder
from sh_encoder import SHEncoder
from positional_encoder import PositionalEmbedder
from small_nerf import NeRFSmall
from integrator import Integrator

# Import modules for testing code
import json
from rays_generator import RaysGenerator
import numpy as np
from torchinfo import summary

# Parameters
from parameters import *


# Model class
class NeuralRenderer(nn.Module):
    def __init__(self,
                 
                 # Sampler args 
                 n_ray_samples: int,
                 near: float, 
                 far: float,
                 
                 # Hash Embedder args
                 bbox: tuple,
                 n_levels: int,
                 n_features_per_level: int,
                 log2_hashmap_size: int, 
                 low_res: int,
                 high_res: int,
                 
                 # Positional Encoder args
                 n_freq: int,

                 # SH Encoder args
                 input_dim: int,
                 degree: int,
                 out_dim: int,

                 # NeRF args
                 num_embeddings: int,
                 embedding_dim: int,
                 n_layers: int,
                 hidden_dim: int,
                 geo_feat_dim: int,
                 n_layers_color: int,
                 hidden_dim_color: int,
                 input_ch: int,
                 input_ch_views: int,
                 out_ch: int,

                 # Target device
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
        
        # Generate encoding for view directions (SH or Positional Encoding)
        self.sh_encoder = SHEncoder(input_dim=input_dim,
                                    degree=degree,
                                    out_dim=out_dim,
                                    device=device)
        
        self.pos_encoder = PositionalEmbedder(n_freq=n_freq)
        
        # Infer the density and channel values for each sample along a ray
        self.nerf = NeRFSmall(num_embeddings=num_embeddings,
                              embedding_dim=embedding_dim,
                              n_layers=n_layers,
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


    def forward(
            self,
            rays: torch.Tensor,
            app_code: int
    ):

        # Bring rays to target device
        cam_rays = rays[:, :6].to(self.device)
        app_code = rays[:, 6].int().to(self.device)

        # Preprocessing steps are carried out without retaining the gradient
        with torch.no_grad():

            # Produce samples along each ray produced from camera
            samples, zvals = self.sampler(cam_rays[..., :3], cam_rays[..., 3:6])
            samples = samples.to(self.device)
            zvals = zvals.to(self.device)

            # Concatenate points with view direction and kill one dimension
            viewdirs = cam_rays[..., 3:6].unsqueeze(1).to(self.device)
            app_code = app_code[..., 6].unsqueeze(1).to(self.device)

            viewdirs = viewdirs.expand(-1, samples.shape[1], -1)
            app_code = app_code.expand(-1, samples.shape[1], -1)

            rays_and_viewdirs = torch.cat([samples, viewdirs], dim=-1)
            rays_and_viewdirs = rays_and_viewdirs.reshape(-1, 6)

        # Hash-encode the samples coordinates and SH-encode the view directions
        enc_points, keep_mask = self.embedder(rays_and_viewdirs[..., :3])
        #enc_points = self.pos_encoder(rays_and_viewdirs[..., :3])
        enc_dirs = self.sh_encoder(rays_and_viewdirs[..., 3:6])
        #enc_dirs = self.pos_encoder(rays_and_viewdirs[..., 3:6])
        
        # Concatenate as a whole input vector and compute a forward pass with SmallNeRF
        input_vector = torch.cat([enc_points, enc_dirs], dim=-1).to(self.device)
        output = self.nerf(input_vector, app_code)

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
        calib[:2, :3] /=  1024./cfg_parameters["resolution"]

    raygen = RaysGenerator(**rays_parameters, K=calib)
    model = NeuralRenderer(**sampler_parameters,
                           **posenc_parameters,
                           **sh_parameters,
                           **hash_parameters,
                           **nerf_parameters,
                           device="cpu")
    
    print(summary(model))