# =======================================================================================
# IMPORT MODULES
# =======================================================================================
import torch
import torch.nn as nn

# =======================================================================================
# IMPORT CUSTOM MODULES
# =======================================================================================
from model.sampler import Sampler
from model.hash_embedder import HashEmbedder
from model.sh_encoder import SHEncoder
from model.appearance_embedder import AppearanceEmbedder
from model.small_nerf import NeRFSmall
from model.integrator import Integrator
from model.occupancy_grid import OccupancyGrid

# =======================================================================================
# IMPORT GLOBAL HYPERPARAMETERS
# =======================================================================================
from configs.config_parameters import *


class NeuralRenderer(nn.Module):
    """
    This class wraps the full NeRF pipeline, from sampling to the image formation (integration).
    """
    def __init__(self,
                 n_ray_samples: int, near: float, far: float, bbox: tuple,
                 n_levels: int, n_features_per_level: int, log2_hashmap_size: int, 
                 low_res: int, high_res: int, input_dim: int, degree: int,
                 embeddings_num: int, embeddings_dim: int, n_layers: int,
                 hidden_dim: int, geo_feat_dim: int, n_layers_color: int,
                 hidden_dim_color: int, input_ch: int, input_ch_views: int,
                 out_ch: int, min_visible_levels: int, 
                 grid_resolution: int = 128, device: str = 'cpu',
                 level_warmup_steps: int = 0,
                 level_warmup_start_epoch: int = 0,
                 enable_coarse_to_fine: bool = False,
                 enable_hierarchical_sampling: bool = False,
                 n_importance_samples: int = 0,
                 importance_eps: float = 1e-5,
                 enable_occupancy_grid: bool = False,
                 occupancy_threshold: float = 0.01,
                 **kwargs):
        
        super(NeuralRenderer, self).__init__()

        # Model hyperparameters and structural attributes
        self.device = torch.device(device)
        self.out_ch = out_ch  # Represents target image channels (e.g., 1 for IR, 3 for RGB)
        self.enable_hierarchical_sampling = bool(enable_hierarchical_sampling)
        self.n_importance_samples = max(0, int(n_importance_samples))
        self.importance_eps = float(importance_eps)
        
        # Initialize rendering and sampling sub-components
        self.sampler = Sampler(n_samples=n_ray_samples, near=near, far=far, device=device)
        
        self.embedder = HashEmbedder(bbox=bbox, n_levels=n_levels, n_features_per_level=n_features_per_level,
                                     log2_hashmap_size=log2_hashmap_size, low_resolution=low_res,
                                     high_resolution=high_res, device=device, min_visible_levels=min_visible_levels,
                                     level_warmup_steps=level_warmup_steps,
                                     level_warmup_start_epoch=level_warmup_start_epoch,
                                     enable_coarse_to_fine=enable_coarse_to_fine)
        
        self.sh_encoder = SHEncoder(input_dim=input_dim, degree=degree, device=device)
        self.app_embedder = AppearanceEmbedder(embeddings_num=embeddings_num, embeddings_dim=embeddings_dim, device=device)

        # Instantiate NeRF network mapping coordinates and view vectors to radiance fields
        self.nerf = NeRFSmall(n_layers=n_layers, hidden_dim=hidden_dim, geo_feat_dim=geo_feat_dim,
                              n_layers_color=n_layers_color, embeddings_dim=embeddings_dim,
                              hidden_dim_color=hidden_dim_color, input_ch=input_ch,
                              input_ch_views=input_ch_views, out_ch=out_ch, device=device).to(device)
        
        self.integrator = Integrator(device=device)

        # Optional occupancy grid for empty-space skipping. It is only created when
        # enabled, so checkpoints trained without it load unchanged. Skipping starts
        # once the trainer has updated the grid at least once (occupancy_active).
        self.occupancy_threshold = float(occupancy_threshold)
        self.occupancy_grid = (
            OccupancyGrid(bbox=bbox, resolution=grid_resolution, device=device)
            if enable_occupancy_grid else None
        )
        self.register_buffer("occupancy_active", torch.zeros((), dtype=torch.bool), persistent=enable_occupancy_grid)

    @torch.no_grad()
    def update_occupancy_grid(self, density_threshold: float = None) -> None:
        """Re-evaluate the density on the grid and mark empty voxels (no-op when disabled)."""
        if self.occupancy_grid is None:
            return
        threshold = self.occupancy_threshold if density_threshold is None else density_threshold
        self.occupancy_grid.update_grid(self.embedder, self.nerf, density_threshold=threshold)
        self.occupancy_active.fill_(True)

    def occupancy_mask(self, flat_samples: torch.Tensor):
        """Boolean mask of samples in occupied space, or None when skipping is off."""
        if self.occupancy_grid is None or not bool(self.occupancy_active):
            return None
        return self.occupancy_grid.query_mask(flat_samples, threshold=self.occupancy_threshold)

    def query_nerf(self, nerf_input: torch.Tensor, occupied: torch.Tensor = None) -> torch.Tensor:
        """
        Run the NeRF MLPs, only on occupied samples when a mask is given. Skipped samples
        get zero colour and zero density, so they contribute nothing to the integral.
        """
        if occupied is None:
            return self.nerf(nerf_input)
        output = self.nerf(nerf_input[occupied])
        full = output.new_zeros((nerf_input.shape[0], output.shape[-1]))
        full[occupied] = output
        return full

    def forward(
            self,
            rays: torch.Tensor,
            app_idx: torch.Tensor,
            epoch: int = None,
            n_iter: int = None,
            max_steps: int = None,
            len_dl: int = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sequence of functions turning input rays into a rendered pixel map, depth map, 
        sparsity loss, and segmentation mask.
        """
        rays = rays.to(self.device)
        app_idx = app_idx.to(torch.int32).to(self.device)
        app_emb = self.app_embedder(app_idx).squeeze(1) if self.app_embedder(app_idx).dim() == 3 else self.app_embedder(app_idx)

        # Produce coarse samples; optional PDF resampling is performed after the
        # coarse pass using detached FP32 weights.
        samples, zvals = self.sampler(rays[..., :3], rays[..., 3:6])
        n_rays, n_samples, _ = samples.shape

        # Dynamic channel setup: target image channels + 1 volume density (sigma) channel
        total_network_out_ch = self.out_ch + 1

        # -------------------------------------------------------------------------------
        # 1. EMBEDDING ON ALL SAMPLING POINTS
        # -------------------------------------------------------------------------------
        current_step = (epoch * len_dl) + n_iter if all(v is not None for v in [epoch, n_iter, len_dl]) else None

        # Flatten points array to interface smoothly with the spatial hash embedder
        flat_samples = samples.reshape(-1, 3)
        enc_points, keep_mask = self.embedder(
            coords=flat_samples,
            current_step=current_step,
            max_steps=max_steps,
            steps_per_epoch=len_dl,
        )
        enc_points = enc_points * keep_mask.unsqueeze(-1).float()

        # -------------------------------------------------------------------------------
        # 2. ENCODE DIRECTIONS & APPEARANCE FOR ALL SAMPLES
        # -------------------------------------------------------------------------------
        enc_dirs = self.sh_encoder(rays[..., 3:6])  # Shape: [n_rays, dir_dim]
        
        # Stretch and repeat ray directions and latent codes for all sampling spaces along the ray
        enc_dirs_stretched = enc_dirs.unsqueeze(1).repeat(1, n_samples, 1).reshape(-1, enc_dirs.shape[-1])
        app_emb_stretched = app_emb.unsqueeze(1).repeat(1, n_samples, 1).reshape(-1, app_emb.shape[-1])

        # -------------------------------------------------------------------------------
        # 3. NETWORK FORWARDPASS
        # -------------------------------------------------------------------------------
        nerf_input = torch.cat([enc_points, enc_dirs_stretched, app_emb_stretched], dim=-1)
        output_flat = self.query_nerf(nerf_input, self.occupancy_mask(flat_samples))

        # Unflatten network outputs using the dynamic channel count
        nerf_output = output_flat.reshape(n_rays, n_samples, total_network_out_ch)

        # -------------------------------------------------------------------------------
        # 4. VOLUMETRIC INTEGRATION
        # -------------------------------------------------------------------------------
        chs_map, depth_map, sparsity_loss, mask = self.integrator(nerf_output, zvals, rays[..., 3:6])
        if self.enable_hierarchical_sampling and self.n_importance_samples > 0:
            fine_z = self.sampler.sample_pdf(
                zvals, self.integrator.last_weights.detach(),
                self.n_importance_samples, self.importance_eps)
            zvals = torch.cat([zvals.float(), fine_z], dim=-1).sort(dim=-1).values
            samples = rays[..., None, :3] + rays[..., None, 3:6] * zvals[..., :, None]
            n_samples = zvals.shape[-1]
            flat_samples = samples.reshape(-1, 3)
            enc_points, keep_mask = self.embedder(flat_samples, current_step, max_steps, len_dl)
            enc_points = enc_points * keep_mask.unsqueeze(-1).float()
            enc_dirs_stretched = enc_dirs.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, enc_dirs.shape[-1])
            app_emb_stretched = app_emb.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, app_emb.shape[-1])
            output_flat = self.query_nerf(torch.cat([enc_points, enc_dirs_stretched, app_emb_stretched], dim=-1),
                                          self.occupancy_mask(flat_samples))
            nerf_output = output_flat.reshape(n_rays, n_samples, total_network_out_ch)
            chs_map, depth_map, sparsity_loss, mask = self.integrator(nerf_output, zvals, rays[..., 3:6])
        
        return chs_map, depth_map, sparsity_loss, mask