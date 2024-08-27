# Import modules
import torch
import torch.nn as nn

# Import custom modules
from parameters import RenderingParameters
from rays_generator import RaysGenerator
from sampler import Sampler
from embedder import HashEmbedder
from sh_encoder import SHEncoder
from small_nerf import NeRFSmall
from integrator import Integrator


class NeuralRenderer(nn.Module):
    def __init__(self,
                 H: int, W: int, CH: int, K: int,
                 n_ray_samples: int, near: float, far: float,
                 bbox: tuple, n_levels: int, n_features_per_level: int, log2_hashmap_size: int, low_res: int, high_res: int, device: str,
                 input_dim: int, degree: int, out_dim: int,
                 n_layers: int, hidden_dim: int, geo_feat_dim: int, n_layers_color: int, hidden_dim_color: int, input_ch: int, input_ch_views: int, out_ch: int):
        super(NeuralRenderer, self).__init__()
        
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
        
        # Generate encoding for view directions
        self.sh_encoder = SHEncoder(input_dim=input_dim,
                                    degree=degree,
                                    out_dim=out_dim)
        
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
        self.integrator = Integrator()


    def forward(self,
                c2w: torch.Tensor,
                frame: torch.Tensor = torch.tensor([])):

        # Preprocessing steps are carried out without retaining the gradient
        with torch.no_grad():

            # If a frame is given, the model is employed in training mode. In essence, it returns
            # both rays (origins, directions) and the corresponding true channel values (RGB+). Vice versa.
            # when it is employed in rendering mode, only rays are provided.
            if frame.shape[0] == 0:
                rays = self.rays_generator(c2w)
                labels = torch.tensor([])
            else:
                rays_and_labels = self.rays_generator(c2w, frame)
                rays = rays_and_labels[..., :6]
                labels = rays_and_labels[..., 6:]

            # Produce samples along each ray produced from camera
            samples, zvals = self.sampler(rays[..., :3], rays[..., 3:6])

            # Concatenate points with view direction and kill one dimension
            viewdirs = rays[..., 3:6].unsqueeze(1)
            viewdirs = viewdirs.expand(-1, samples.shape[1], -1)
            rays_and_viewdirs = torch.cat([samples, viewdirs], dim=-1)
            rays_and_viewdirs = rays_and_viewdirs.reshape(-1, 6)

        # Hash-encode the samples coordinates and SH-encode the view directions
        enc_points, keep_mask = self.embedder(rays_and_viewdirs[..., :3])
        enc_dirs = self.sh_encoder(rays_and_viewdirs[..., 3:6])

        # Concatenate as a whole input vector and compute a forward pass with NeRF
        input_vector = torch.cat([enc_points, enc_dirs], dim=-1)
        output = self.nerf(input_vector)

        # Clean (i.e. set sigma to 0) output estimates out of bbox boundaries
        # and reshape as [points, samples, channels]
        output[~keep_mask, -1] = 0
        output = output.reshape(rays.shape[0], self.sampler.n_samples, -1)
        
        # Integrate densities and channels values estimate along each ray
        chs_map, depth_map, sparsity_loss = self.integrator(output, zvals, rays[..., 3:6])

        return chs_map, depth_map, sparsity_loss, labels
         


# Run for usage example
if __name__ == "__main__":

    # Import parameters
    params_obj = RenderingParameters()
    params = params_obj.get_all_params()

    # Instantiate the model object
    renderer = NeuralRenderer(**params)

    # Generate dummy inputs
    dummy_c2w = torch.rand((4, 4), dtype=torch.float32)

    # Infer
    frame, depth, sparsity, labels = renderer(dummy_c2w)
    print(frame.shape)
