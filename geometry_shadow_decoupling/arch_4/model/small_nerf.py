# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRFSmall(nn.Module):
    """
    This class implements the core of the NeRF model. It separates geometry (sigma)
    from look-dependent appearance (color) using two sequential MLP branches.
    """
    def __init__(
            self,
            n_layers: int,
            hidden_dim: int,
            geo_feat_dim: int,
            embeddings_dim: int,
            n_layers_color: int,
            hidden_dim_color: int,
            input_ch: int,
            input_ch_views: int,
            out_ch: int,
            device: str = 'cpu'
    ):
        super(NeRFSmall, self).__init__()

        # Attributes
        self.device = torch.device(device)
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.input_app_dim = embeddings_dim
        self.n_layers_color = n_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.out_ch = out_ch

        # Build networks
        self.sigma_net = self.build_sigma_net()
        self.color_net = self.build_color_net()

    def build_sigma_net(self) -> nn.ModuleList:
        sigma_net = list()
        for layer in range(self.n_layers):
            in_dim = self.input_ch if layer == 0 else self.hidden_dim
            out_dim = (1 + self.geo_feat_dim) if layer == self.n_layers - 1 else self.hidden_dim
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        return nn.ModuleList(sigma_net)

    def build_color_net(self) -> nn.ModuleList:
        color_net = list()
        for layer in range(self.n_layers_color):
            in_dim = (self.input_ch_views + self.geo_feat_dim + self.input_app_dim) if layer == 0 else self.hidden_dim_color
            out_dim = self.out_ch if layer == self.n_layers_color - 1 else self.hidden_dim_color
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        return nn.ModuleList(color_net)
    
    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """
        Inference of NeRF model. Maps position and view/appearance to color and raw density.
        """
        # Ensure tensor is on the correct device
        rays = rays.to(self.device)
        
        # Split features chunks safely from the processed input vector
        input_pts, input_views, input_app = torch.split(
            rays,
            [self.input_ch, self.input_ch_views, self.input_app_dim],
            dim=-1
        )
        
        # --- 1. SIGMA BRANCH (GEOMETRY) ---
        out = input_pts
        for layer_idx, layer in enumerate(self.sigma_net):
            out = layer(out)
            # Applica ReLU solo sui livelli intermedi; l'ultimo livello deve rimanere lineare
            if layer_idx < self.n_layers - 1:
                out = F.relu(out, inplace=True)

        # Extraction of unconstrained raw sigma and geometric features
        raw_sigma, geo_features = out[..., 0], out[..., 1:]
        
        # --- 2. COLOR BRANCH (RADIANCE) ---
        color = torch.cat([input_views, geo_features, input_app], dim=-1)
        for layer_idx, layer in enumerate(self.color_net):
            color = layer(color)

            if layer_idx != self.n_layers_color - 1:
                color = F.relu(color, inplace=True)
            else:
                color = torch.sigmoid(color)
            
        # Pack everything together: [Color (RGB), Raw_Sigma (1)]
        outputs = torch.cat([color, raw_sigma.unsqueeze(-1)], dim=-1)
        
        return outputs