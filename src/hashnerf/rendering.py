# Import modules
import torch
import torch.nn as nn

# Import custom modules
from rays_generator import RaysGenerator
from small_nerf import NeRFSmall


class NeuralRenderer(nn.Module):
    def __init__(self,
                 H: int,
                 W: int,
                 CH: int,
                 K: int,
                 mode: str,
                 ):
        super(NeuralRenderer).__init__()

        # Attributes

        # Actors
        self.rays_generator = RaysGenerator(H=H, W=W, CH=CH, K=K, mode=mode)