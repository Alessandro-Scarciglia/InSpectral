# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class NeRFSmall(nn.Module):
    def __init__(self,
                 n_layers: int = 3,
                 hidden_dim: int = 64,
                 geo_feat_dim: int = 15,
                 n_layers_color: int = 4,
                 hidden_dim_color: int = 64,
                 input_ch: int = 3,
                 input_ch_views: int = 3,
                 out_ch: int = 3,
                 device: str = 'cpu'):
        super(NeRFSmall, self).__init__()

        # Attributes
        self.device = torch.device(device)
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.n_layers_color = n_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.out_ch = out_ch

        # Build sigma network
        self.sigma_net = self.build_sigma_net()

        # Build color network
        self.color_net = self.build_color_net()


    def build_sigma_net(self):
        '''
        Build the network which estimates density along rays.
        '''

        # Initialize list of layers
        sigma_net = list()

        # Build layer by layer
        for layer in range(self.n_layers):
        
            # If it is the first layer, set input channel dimension.
            # Else, set the hidden dimension.
            if layer == 0:
                in_dim = self.input_ch
            else:
                in_dim = self.hidden_dim

            # If it is the last layer set dimension 1 (Sigma) plus
            # 15 (SH color features). Else, set hidden dimension.
            if layer == self.n_layers - 1:
                out_dim = 1 + self.geo_feat_dim
            else:
                out_dim = self.hidden_dim

            # Append the layer 
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # Build the network model as a ModuleList
        sigma_net_model = nn.ModuleList(sigma_net)

        return sigma_net_model


    def build_color_net(self):
        '''
        Build the network which estimates the output color.
        '''

        # Initialize the list of layers
        color_net = list()

        # Build layer by layer
        for layer in range(self.n_layers_color):

            # If it is the first layer set it to input channel dimension
            # plus the SH encoding dimension. Else, set it to hidden dimension.
            if layer == 0:
                in_dim = 2 * self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color

            # If it is the last layer, set it to output channel dimension.
            # Else, set it to hidden dimension.
            if layer == self.n_layers_color - 1:
                out_dim = self.out_ch
            else:
                out_dim = self.hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # Build the network model as a ModuleList
        color_net_model = nn.ModuleList(color_net)

        return color_net_model
    

    def forward(self,
                rays: torch.TensorType):
        '''
        Inference method.
        '''

        # Bring rays to target device
        rays = rays.to(self.device)
        
        # Split origin
        input_pts, input_views, input_sundir = torch.split(rays,
                                                           [self.input_ch, self.input_ch_views, self.input_ch_views],
                                                           dim=-1)
        
        # Sigma estimation branch: usually only geometric points (x, y, z) are necessary to estimate the
        # volumetric density. Indeed, the volumetric density of a point should not be linked to the viewdir.
        # However, including also viewdir in input, the PSNR improves by 1.5pt ca. at the first epoch. Overfitting?
        out = input_pts
        for layer in range(self.n_layers):
            out = self.sigma_net[layer](out)
            out = F.relu(out, inplace=True)

        # Extraction of sigma and geo features
        sigma, geo_features = out[..., 0], out[..., 1:]
        
        # Color estimation branch
        out = torch.cat([input_views, input_sundir, geo_features], dim=-1)
        for layer in range(self.n_layers_color):
            out = self.color_net[layer](out)

            # If the layer is not the last, add relu unit
            if layer != self.n_layers_color - 1:
                out = F.relu(out, inplace=True)
        
        # Extract color and produce inference output
        color = out
        outputs = torch.cat([color, sigma.unsqueeze(-1)], dim=-1)
        
        return outputs



# Usage Test
if __name__ == "__main__":
    
    # Instantiate the model object
    model = NeRFSmall(device="cuda").to("cuda")