# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.profiler import timing_decorator


class NeRFSmall(nn.Module):
    """
    This class implements the core of the NeRF model. In essence, a NeRF model is made of two subsequent branches: the
    former estimates the geometry of the scene; the latter estimate colors per each sample scene. This model does not render
    the final image, since this task is covered by the Integrator class. Moreover, this specific model also employs a third 
    small MLP to model an appearance embedding as a function of the light direction.

    Attributes:
    ----------
    n_layers: int
        it is the number of MLP layers in the sigma network (geometry estimation).
    hidden_dim: int
        it is the number of perceptron in each hidden layer (geometry estimation).
    n_layers_light: int
         it is the number of MLP layers in the light network (appearance estimation).
    hidden_dim_light: int
        it is the number of perceptron in each hidden layer (appearance estimation).
    geo_feat_dim: int
        it is the dimension of the geometric features, that is the output of the first MLP (sigma network) which is
        passed as input to the next MLP (color network).
    n_layers_color: int
        it is the number of MLP layers in the color network (color estimation).
    hidden_dim_color: int
        it is the number of perceptron in each hidden layer (color estimation).
    input_ch: int
        it depends on the coordinates embeddings. It can be computed like n_levels * n_feature_per_level in
        the specific case of Hash Embedding.
    input_ch_views_ch: int
        it is the same as the 'out_dim' dimension selected for the SH encoding of the vieweing direction.
    out_ch: int
        it is the number of channels of the output image (1 for grayscale, 3 for RGB, N for custom stacks).
    device: str
            it is the target device where to move the computation ("cpu" by default, generic "cuda" or specific "cuda:x").
    """
    def __init__(
            self,
            n_layers: int,
            hidden_dim: int,
            n_layers_light: int,
            hidden_dim_light: int,
            geo_feat_dim: int,
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

    def build_sigma_net(
            self
    ) -> nn.ModuleList:
        """
        This method builds the network architecture for the geometry estimation.

        Returns:
        -------
        sigma_net_model: nn.ModuleList
            it is a ModuleList object which contains all the layers of the designed MLP.
        """

        # Initialize list of layers
        sigma_net = list()

        # Build the network layer by layer
        for layer in range(self.n_layers):
        
            # If it is the first layer, set input channel dimension, otherwise set the hidden dimension.
            if layer == 0:
                in_dim = self.input_ch
            else:
                in_dim = self.hidden_dim

            # If it is the last layer set dimension 1 (sigma) plus the 'out_dim' of SH. Otherwise, set hidden dimension.
            if layer == self.n_layers - 1:
                out_dim = 1 + self.geo_feat_dim
            else:
                out_dim = self.hidden_dim

            # Append the layer 
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # Build the network model as a ModuleList
        sigma_net_model = nn.ModuleList(sigma_net)

        return sigma_net_model

    def build_color_net(
            self
    ) -> nn.ModuleList:
        """
        This method builds the network architecture for the color estimation.

        Returns:
        -------
        color_net_model: nn.ModuleList
            it is a ModuleList object which contains all the layers of the designed MLP.
        """

        # Initialize the list of layers
        color_net = list()

        # Build layer by layer
        for layer in range(self.n_layers_color):

            # If it is the first layer set it to input channel dimension plus the SH encoding dimension.
            # Otherwise, set it to hidden dimension.
            if layer == 0:
                in_dim = 2 * self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color

            # If it is the last layer, set it to output channel dimension, otherwise set it to hidden dimension.
            if layer == self.n_layers_color - 1:
                out_dim = self.out_ch
            else:
                out_dim = self.hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # Build the network model as a ModuleList
        color_net_model = nn.ModuleList(color_net)

        return color_net_model
    
    def build_light_net(
            self
    ) -> nn.ModuleList:
        """
        This method builds the network architecture for the appearance estimation.

        Returns:
        -------
        color_net_model: nn.ModuleList
            it is a ModuleList object which contains all the layers of the designed MLP.
        """

        # Initialize the list of layers
        light_net = list()

        # Build the network layer by layer
        for layer in range(self.n_layers_light):

            # If it is the first layer set it to input channel dimension plus the SH encoding dimension.
            # Otherwise, set it to hidden dimension.
            if layer == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_light

            # If it is the last layer, set it to output channel dimension, otherwise set it to hidden dimension.
            if layer == self.n_layers_light - 1:
                out_dim = self.input_ch_views
            else:
                out_dim = self.hidden_dim_light

            light_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # Build the network model as a ModuleList
        light_net_model = nn.ModuleList(light_net)

        return light_net_model
    
    #@timing_decorator
    def forward(
            self,
            rays: torch.TensorType
    ) -> torch.Tensor:
        """
        This method implement the inference of NeRF model. It is fed with a chunk of rays (rays_0, rays_d). Then, a first 
        estimation is made by the sigma network, which exploits input coordinates (after being hash-embedded) to produce an
        estimate of both the volumetric density of each particle (somehow, a score which is 0 in empty spaces and very high where
        the object is placed) and geometric features of such particles.

        The light network models the appearance of the scene, thus decoupling the geometry from any light/shadow effect. It takes in
        input the light direction and produces an appearance embedding. 

        The color network estimate the color of each particle along the input rays. It employes the geometric features, the 
        appearance embedding and the viewing direction.

        Parameters:
        ----------
        rays: torch.Tensor[float]
            it is a stacked version of rays origin and rays direction.
    
        Returns:
        -------
        outputs: torch.Tensor[float]
            it is the color estimate for each channel for each sample along each ray.
        """

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
