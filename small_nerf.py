# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class NeRFSmall(nn.Module):
    def __init__(self,
                 n_layers: int = 3,
                 hidden_dim: int = 64,
                 n_layers_light: int = 3,
                 hidden_dim_light: int = 128,
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
        self.n_layers_light = n_layers_light
        self.hidden_dim_light = hidden_dim_light
        self.geo_feat_dim = geo_feat_dim
        self.n_layers_color = n_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.out_ch = out_ch

        # Build sigma network
        self.sigma_net = self.build_sigma_net()

        # Build color network
        self.color_net = self.build_color_net()

        # Build light network
        self.light_net = self.build_light_net()


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
    

    def build_light_net(self):
        '''
        Build the network which estimates the light intensity.
        '''

        # Initialize the list of layers
        light_net = list()

        # Build layer by layer
        for layer in range(self.n_layers_light):

            # If it is the first layer set it to input channel dimension
            # plus the SH encoding dimension. Else, set it to hidden dimension.
            if layer == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_light

            # If it is the last layer, set it to output channel dimension.
            # Else, set it to hidden dimension.
            if layer == self.n_layers_light - 1:
                out_dim = self.input_ch_views
            else:
                out_dim = self.hidden_dim_light

            light_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # Build the network model as a ModuleList
        light_net_model = nn.ModuleList(light_net)

        return light_net_model
    

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
        
        # Sigma estimation branch
        out = input_pts
        for layer in range(self.n_layers):
            out = self.sigma_net[layer](out)
            out = F.relu(out, inplace=True)

        # Extraction of sigma and geo features
        sigma, geo_features = out[..., 0], out[..., 1:]

        # Light estimation branch
        geo_features_detached = geo_features.detach()
        fading_coeff = torch.cat([input_sundir, geo_features_detached], dim=-1)
        for layer in range(self.n_layers_light):
            fading_coeff = self.light_net[layer](fading_coeff)
            
            if layer != self.n_layers_color - 1:
                fading_coeff = F.relu(fading_coeff)
        
        # Color estimation branch
        color = torch.cat([input_views, fading_coeff, geo_features], dim=-1)
        for layer in range(self.n_layers_color):
            color = self.color_net[layer](color)

            # If the layer is not the last, add relu unit
            if layer != self.n_layers_color - 1:
                color = F.relu(color, inplace=True)
            else:
                color = F.sigmoid(color)
            
        outputs = torch.cat([color, sigma.unsqueeze(-1)], dim=-1)
        
        return outputs


# Usage Test
if __name__ == "__main__":
    
    # Instantiate the model object
    model = NeRFSmall(device="cuda").to("cuda")