# Import modules
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.distributions import Categorical
from config_parameters import *
from model.profiler import timing_decorator


class Integrator(nn.Module):
    """
    This class implements the integration of all estimates made along the say directions (per each sample)
    and computes the transmittance integral to retrieve the value of the corresponding pixel.

    Attributes:
    ----------
    device: str
        it is the target device for computation ("cpu" by default or "cuda", or even a specific GPU "cuda:x")
    """
    def __init__(
            self,
            device : str = 'cpu'
    ):
        super(Integrator, self).__init__()

        # Attributes
        self.device = torch.device(device)

    #@timing_decorator
    def forward(
            self,
            raw: torch.Tensor,
            zvals: torch.Tensor,
            rays_d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        It computes the accumulated transmittance along each ray.

        Parameters:
        ----------
        raw: torch.Tensor[float]
            it is the output of NeRF model inference. It is already filtered with a keep_mask, to zero
            all the estimates made out of the rendering volume.
        zvals: torch.Tensor[float]
            it represents the depth of each sample along the ray, for each ray in the batch.
        rays_d: torch.Tensor[float]
            it is the collection of all the batch ray directions in the rendering scene.

        Returns:
        -------
        chs_map: torch.Tensor[float]
            it is the tensor containing all the channels estimates of the batch pixels.
        depth_map: torch.Tensor[float]
            it is the depthmap in 0-255 where 0 corresponds to 'near' and 255 to 'far' values set in the 
            configuration file. It is the distance from the camera origin at which sampling starts/ends. To have a 
            consistent metrics, it is suggested to fix same boundaries as in Blender simulation [8-16].
        sparsity_loss: torch.Tensor[float]
            it is a loss which penalizes uniformity in weights along samples (based on Information Entropy principle).
        mask: torch.Tensor[float]
            the segmentation mask obtained from volume density estimates.
        '''

        # Bring inputs to target device
        raw = raw.to(self.device)
        zvals = zvals.to(self.device)
        rays_d = rays_d.to(self.device)

        # Define a lambda function to compute alphas from samples
        raw2alpha = lambda raw, dists, act_fn = relu: 1. - torch.exp(-act_fn(raw) * dists)

        # Compute distances and reshape as [rays, samples]
        dists = zvals[..., 1:] - zvals[..., :-1]
        dists = torch.cat([dists, torch.full(dists[..., :1].shape, 1e10, device=self.device)], dim=-1)
        dists *= torch.norm(rays_d[..., None, :], dim=-1)

        # Extracts channes values 
        chs = raw[..., :cfg_parameters["channels"]]

        # Compute alphas and cumulative product
        alpha = raw2alpha(raw[..., -1], dists)
        tmp = torch.cat([torch.ones((alpha.shape[0], 1), device=self.device), 1. - alpha + 1e-10], dim=-1)
        cumprod = torch.cumprod(tmp, dim=-1)[:, :-1]

        # Compute integration weights and channels values, as [rays, chs]
        weights = alpha * cumprod
        mask = torch.clamp(torch.sum(weights, dim=-1), min=0.0, max=1.1).unsqueeze(-1)
        chs_map = torch.sum(weights[..., None] * chs, dim=-2)

        # Compute integration of weights and densities for depth, as [rays, depth]
        zvals = (sampler_parameters["far"] - zvals) / (sampler_parameters["far"] - sampler_parameters["near"])
        depth_map = torch.sum(weights * zvals, dim=-1) / (torch.sum(weights, dim=-1) + 1e-5)

        # Finally, compute weights sparsity loss
        try:
            sparsity_loss = Categorical(
                probs = torch.cat([weights, 1.0 - weights.sum(-1, keepdim=True) + 1e-6], dim=-1)
            ).entropy().sum()
        except:
            sparsity_loss = torch.tensor(0.0, device=self.device)
            print("Warning: Sparsity Loss cannot be computed.")

        return chs_map, depth_map, sparsity_loss, mask
