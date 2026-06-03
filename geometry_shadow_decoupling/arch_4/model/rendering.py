# Import modules
import torch
import torch.nn as nn

# Import custom modules
from model.sampler import Sampler
from model.hash_embedder import HashEmbedder
from model.sh_encoder import SHEncoder
from model.appearance_embedder import AppearanceEmbedder
from model.small_nerf import NeRFSmall
from model.integrator import Integrator

# Parameters
from config_parameters import *


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
                 out_ch: int, device: str = 'cpu'):
        
        super(NeuralRenderer, self).__init__()

        # Attributes
        self.device = torch.device(device)
        
        self.sampler = Sampler(n_samples=n_ray_samples, near=near, far=far, device=device)
        
        self.embedder = HashEmbedder(bbox=bbox, n_levels=n_levels, n_features_per_level=n_features_per_level,
                                     log2_hashmap_size=log2_hashmap_size, low_resolution=low_res,
                                     high_resolution=high_res, device=device)
        
        self.sh_encoder = SHEncoder(input_dim=input_dim, degree=degree, device=device)
        
        self.app_embedder = AppearanceEmbedder(embeddings_num=embeddings_num, embeddings_dim=embeddings_dim, device=device)

        self.nerf = NeRFSmall(n_layers=n_layers, hidden_dim=hidden_dim, geo_feat_dim=geo_feat_dim,
                              n_layers_color=n_layers_color, embeddings_dim=embeddings_dim,
                              hidden_dim_color=hidden_dim_color, input_ch=input_ch,
                              input_ch_views=input_ch_views, out_ch=out_ch, device=device).to(device)
        
        self.integrator = Integrator(near=near, far=far, device=device)

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
        # Bring basic inputs to target device immediately
        rays = rays.to(self.device)
        app_idx = app_idx.to(torch.int32).to(self.device)
        
        # Extract appearance embedding
        app_emb = self.app_embedder(app_idx)  # Shape: [B, 1, emb_dim] o [B, emb_dim]

        # Produce 3D point samples along each camera ray
        samples, zvals = self.sampler(rays[..., :3], rays[..., 3:6])
        
        n_rays, n_samples, _ = samples.shape

        # --- EFFICIENZA: EVITIAMO IL MEGA-RESHAPE PREMATURO ---
        # Estraiamo le coordinate dei punti pronte per l'Hash Embedder
        flat_points = samples.reshape(-1, 3)

        # Gestione flessibile e robusta dello step temporale per il Coarse-to-Fine
        if epoch is not None and n_iter is not None and len_dl is not None:
            current_step = (epoch * len_dl) + n_iter
        else:
            current_step = None  # Indica all'embedder di usare 100% dei livelli (Inferenza/Eval)

        # Codifica spaziale dei punti
        enc_points, keep_mask = self.embedder(coords=flat_points,
                                              current_step=current_step,
                                              max_steps=max_steps)

        # Codifica direzionale (SH): la calcoliamo una volta per raggio e poi la espandiamo
        # Invece di fare l'expand su 6+dim prima del forward, lo facciamo in modo efficiente qui
        viewdirs = rays[..., 3:6] # Shape: [B, 3]
        enc_dirs_per_ray = self.sh_encoder(viewdirs) # Shape: [B, SH_dim]
        
        # Espansione dei vettori condizionali alla cardinalità dei campioni (n_samples)
        enc_dirs = enc_dirs_per_ray.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, enc_dirs_per_ray.shape[-1])
        
        # Uniformiamo la dimensionalità di app_emb se non ha la dimensione del batch corretta
        if app_emb.dim() == 3: 
            app_emb = app_emb.squeeze(1) # Rende la forma [B, emb_dim]
        enc_app = app_emb.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, app_emb.shape[-1])

        # Concatenazione finale ottimizzata direttamente nello spazio latente codificato
        input_vector = torch.cat([enc_points, enc_dirs, enc_app], dim=-1)
        output = self.nerf(input_vector)

        # Applicazione della maschera della scatola limite (Bounding Box Keep Mask)
        output[~keep_mask, -1] = 0.0
        
        # Riordiniamo il tensore nella griglia [Raggi, Campioni, Canali_Uscita]
        output = output.reshape(n_rays, n_samples, -1)
        
        # Integrazione volumetrica finale per generare i pixel dell'immagine
        chs_map, depth_map, sparsity_loss, mask = self.integrator(output, zvals, rays[..., 3:6])
        
        return chs_map, depth_map, sparsity_loss, mask