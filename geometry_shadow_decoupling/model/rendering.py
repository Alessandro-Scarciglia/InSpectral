# Import modules
import torch
import torch.nn as nn

# Import custom modules
from model.sampler import Sampler
from model.hash_embedder import HashEmbedder
from model.sh_encoder import SHEncoder
from model.small_nerf import NeRFSmall
from model.integrator import Integrator

# Parameters
from config_parameters import *


class NeuralRenderer(nn.Module):
    """
    This class wraps the full NeRF pipeline, from sampling to the image formation (integration). It takes as input
    the rays full stack and returns channels estimate, depth map and sparsity loss on sigmas distribution in the scene.

    Attributes:
    ----------
    Sampler Args: see details in 'sampler.py'
    Hash Embedder Args: see details in 'hash_embedder.py'
    SH Encoder Args: see details in 'sh_encoder.py'
    NeRF Args: see details in 'small_nerf.py'

    device: str
        it is the target device where to move the computation ("cpu" by default, generic "cuda" or specific "cuda:x").
    """
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

                 # SH Encoder args
                 input_dim: int,
                 degree: int,

                 # NeRF args
                 n_layers: int,
                 hidden_dim: int,
                 geo_feat_dim: int,
                 n_layers_light: int,
                 hidden_dim_light: int,
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
        
        # Generate a set of samples uniformly distributed along the ray, within the [near, far] interval
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
                                    device=device)

        # Infer the density and channel values for each sample along a ray
        self.nerf = NeRFSmall(n_layers=n_layers,
                              hidden_dim=hidden_dim,
                              geo_feat_dim=geo_feat_dim,
                              n_layers_color=n_layers_color,
                              n_layers_light=n_layers_light,
                              hidden_dim_light=hidden_dim_light,
                              hidden_dim_color=hidden_dim_color,
                              input_ch=input_ch,
                              input_ch_views=input_ch_views,
                              out_ch=out_ch,
                              device=device).to(device)
        
        # Integrates all densities and channels values along a ray to get the overall properties of a single pixel
        self.integrator = Integrator(device=device)

    def forward(
            self,
            rays: torch.TensorType,
            sundir: torch.TensorType
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method implements the sequence of functions which transform the input rays and the sun direction vector
        in a full rendered image, along with the scene depth estimation, the sparsity loss and the object mask. 

        Parameters:
        ----------
        rays: torch.Tensor[float]
            it is the stack containing rays origin, directions.
        sundir: torch.Tensor[float]
            it is the unit verctor which defines the direction of the light source.

        Results:
        -------
        chs_map: torch.Tensor[float]
            it is the estimate of output channels for each pixel of the image (not for each sample of the rays,
            but for each pixel since here we also integrate rays!).
        depth_map: torch.Tensor[float]
            it is the estimate of depth encoded in [0-1] or [0-255] in the range [near, far].
        sparsity_loss: torch.Tensor[float]
            it is a measure of the sparsity of the scene filled volume.
        mask: torch.Tensor[float]
            it is the estimate of the segmentation mask for the single object of the scene. It is a raw mask coming from a
            ReLU activation, thus [o, +inf]. It is grounded in [0, 1] directly in the BCEDiceLoss compiutation.
        """
    
        # Bring rays to target device
        rays = rays.to(self.device)
        sundir = sundir.to(self.device)

        # Produce samples along each ray produced from camera
        samples, zvals = self.sampler(rays[..., :3], rays[..., 3:6])
        samples = samples.to(self.device)
        zvals = zvals.to(self.device)

        # Concatenate points with view direction and kill one dimension
        viewdirs = rays[..., 3:6].unsqueeze(1).to(self.device)
        viewdirs = viewdirs.expand(-1, samples.shape[1], -1)

        # Likewise for sun direction
        sundir = sundir.unsqueeze(1).to(self.device)
        sundir = sundir.expand(-1, samples.shape[1], -1)

        rays_and_viewdirs = torch.cat([samples, viewdirs], dim=-1)
        rays_and_viewdirs = rays_and_viewdirs.reshape(-1, 6)
        sundir = sundir.reshape(-1, 3)

        # Hash-encode the samples coordinates and SH-encode the view directions
        enc_points, keep_mask = self.embedder(rays_and_viewdirs[..., :3])
        enc_dirs = self.sh_encoder(rays_and_viewdirs[..., 3:6])
        enc_sundir = self.sh_encoder(sundir)

        # Concatenate as a whole input vector and compute a forward pass with SmallNeRF
        input_vector = torch.cat([enc_points, enc_dirs, enc_sundir], dim=-1).to(self.device)
        output = self.nerf(input_vector)

        # Clean (i.e. set sigma to 0) output estimates out of bbox boundaries
        # and reshape as [points, samples, channels]
        output = output.to(self.device)
        output[~keep_mask, -1] = 0
        output = output.reshape(rays.shape[0], self.sampler.n_samples, -1)
        
        
        # Integrate densities and channels values estimate along each ray
        chs_map, depth_map, sparsity_loss, mask = self.integrator(output, zvals, rays[..., 3:6])

        return chs_map, depth_map, sparsity_loss, mask
