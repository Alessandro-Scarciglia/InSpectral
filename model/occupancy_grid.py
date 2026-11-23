import torch
import torch.nn as nn

class OccupancyGrid(nn.Module):
    """
    Semplificata: Griglia di occupazione 3D per Ray Marching Skipping con decadimento controllato.
    """
    def __init__(self, bbox: tuple, resolution: int = 128, device: str = 'cpu', ema_decay: float = 0.95):
        super().__init__()
        self.device = torch.device(device)
        self.resolution = resolution
        self.ema_decay = ema_decay 
        
        # Gestione immediata della Bounding Box
        bbox_min_val = bbox[0] if isinstance(bbox[0], torch.Tensor) else torch.tensor(bbox[:3])
        bbox_max_val = bbox[1] if isinstance(bbox[1], torch.Tensor) else torch.tensor(bbox[3:])
        
        self.register_buffer("bbox_min", bbox_min_val.to(self.device).float())
        self.register_buffer("bbox_max", bbox_max_val.to(self.device).float())
        self.register_buffer("grid", torch.ones((resolution, resolution, resolution), device=self.device))

    def query_mask(self, pts: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """
        Interroga la griglia: restituisce True solo per i punti dentro la bbox e sopra la soglia.
        """
        inside_bbox = (pts >= self.bbox_min).all(dim=-1) & (pts <= self.bbox_max).all(dim=-1)
        occupied_mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)
        
        if inside_bbox.any():
            valid_pts = pts[inside_bbox]
            # Normalizzazione e mappatura in indici della griglia [0, resolution-1]
            scaled_pts = (valid_pts - self.bbox_min) / (self.bbox_max - self.bbox_min + 1e-5)
            coords = (scaled_pts * self.resolution).long().clamp(0, self.resolution - 1)
            
            voxel_values = self.grid[coords[:, 0], coords[:, 1], coords[:, 2]]
            occupied_mask[inside_bbox] = voxel_values > threshold
            
        return occupied_mask

    @torch.no_grad()
    def update_grid(self, embedder, nerf_model, density_threshold: float = 0.01, max_pts_per_batch: int = 100000):
        """
        Campiona il NeRF sul volume 3D e aggiorna la griglia con decadimento asimmetrico.
        """
        # Salvataggio stati iniziali delle reti
        training_states = (nerf_model.training, embedder.training)
        nerf_model.eval(); embedder.eval()
        
        # Generazione lineare delle coordinate dei voxel [R, R, R, 3] normalizzate in [0, 1]
        lin = torch.linspace(0.5 / self.resolution, 1.0 - 0.5 / self.resolution, self.resolution, device=self.device)
        grid_pts = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), dim=-1).reshape(-1, 3)
        grid_pts = self.bbox_min + grid_pts * (self.bbox_max - self.bbox_min)
        
        # Forward pass a batch sul volume
        sigmas = []
        for i in range(0, grid_pts.shape[0], max_pts_per_batch):
            batch_pts = grid_pts[i : i + max_pts_per_batch]
            enc_pts, _ = embedder(coords=batch_pts, current_step=None, max_steps=None)
            
            # Tensori fittizi per gli input condizionali del NeRF
            dummy_dirs = torch.zeros((batch_pts.shape[0], nerf_model.input_ch_views), device=self.device)
            dummy_app = torch.zeros((batch_pts.shape[0], nerf_model.input_app_dim), device=self.device)
            
            outputs = nerf_model(torch.cat([enc_pts, dummy_dirs, dummy_app], dim=-1))
            sigmas.append(outputs[..., -1])
            
        sigmas = torch.cat(sigmas, dim=0).reshape(self.resolution, self.resolution, self.resolution)
        # Never use a threshold above the mean density, so a young, still faint model does
        # not get its whole volume marked as empty (as in Instant-NGP).
        effective_threshold = min(float(density_threshold), float(sigmas.clamp_min(0).mean()))
        current_status = (sigmas > effective_threshold).float()
        
        # Aggiornamento asimmetrico pulito: se attivo applica l'EMA, altrimenti decade del 60%
        ema_grid = self.grid * self.ema_decay + current_status * (1.0 - self.ema_decay)
        self.grid = torch.where(current_status > 0, ema_grid, self.grid * 0.4)
        
        # Ripristino stati iniziali
        nerf_model.train(training_states[0]); embedder.train(training_states[1])