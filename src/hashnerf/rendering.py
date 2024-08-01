# Import modules
import torch
import torch.nn as nn

# Import custom modules
from rays_generator import RaysGenerator
from sampler import Sampler
from embedder import HashEmbedder
from small_nerf import NeRFSmall
from integrator import *


class NeuralRenderer(nn.Module):
    def __init__(self,
                 H: int, W: int, CH: int, K: int,
                 n_ray_samples: int, near: float, far: float,
                 bbox: tuple, n_levels: int, n_features_per_level: int, log2_hashmap_size: int, low_res: int, high_res: int, device: str,
                 n_layers: int, hidden_dim: int, geo_feat_dim: int, n_layers_color: int, hidden_dim_color: int, input_ch: int, input_ch_views: int, out_ch: int):
        super(NeuralRenderer).__init__()

        # Generates rays origins and directions 
        self.rays_generator = RaysGenerator(H=H,
                                            W=W,
                                            CH=CH,
                                            K=K)
        
        # Generate a set of samples uniformly distributed along the ray,
        # within the [near, far] interval.
        self.sampler = Sampler(n_samples=n_ray_samples,
                               near=near,
                               far=far)
        
        # Generate embeddings of an input coordinate
        self.embedder = HashEmbedder(bbox=bbox,
                                     n_levels=n_levels,
                                     n_features_per_level=n_features_per_level,
                                     log2_hashmap_size=log2_hashmap_size,
                                     low_resolution=low_res,
                                     high_resolution=high_res,
                                     device=device)
        
        # Infer the density and channel values for each sample along a ray
        self.nerf = NeRFSmall(n_layers=n_layers,
                              hidden_dim=hidden_dim,
                              geo_feat_dim=geo_feat_dim,
                              n_layers_color=n_layers_color,
                              hidden_dim_color=hidden_dim_color,
                              input_ch=input_ch,
                              input_ch_views=input_ch_views,
                              out_ch=out_ch)
        
        # Integrates all densities and channels values along a ray to get the 
        # overall properties of a single pixel.
        self.integrator = None


        # Reorder the output of each single inference in order to render the whole frame
        self.collector = None


    def forward(self,
                c2w: torch.Tensor,
                frame: torch.Tensor = torch.tensor([])):

        # Preprocessing steps are carried out without retaining the gradient
        with torch.no_grad():

            # If a frame is given, the model is employed in training mode. In essence, it returns
            # both rays (origins, directions) and the corresponding true channel values (RGB+). Vice versa.
            # when it is employed in rendering mode, only rays are provided.
            if frame.shape[0] == 0:
                self.rays_generator.rendering_mode()
                rays = self.rays_generator(c2w)
                rays_o, rays_d = rays[..., :3], rays[..., 3:]
                labels = torch.tensor([])
            else:
                self.rays_generator.training_mode()
                rays_and_labels = self.rays_generator(c2w, frame)
                rays_o, rays_d = rays_and_labels[..., :3], rays_and_labels[..., 3:6]
                labels = rays_and_labels[..., 6:]

            # Produce samples along each ray produced from camera
            samples = self.sampler(rays_o, rays_d)

        # Infere sigmas and channel values for each sample
        out = self.nerf(samples)
        

        
