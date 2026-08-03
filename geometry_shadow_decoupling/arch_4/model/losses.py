"""
This module contains the custom loss functions used during the training of the NeRF model.
Includes Hashing, Total Variation, Cauchy Sparsity, and Tunable Bimodal Driving Loss.
"""

# Import modules
from math import exp, log, floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_parameters import *


# Hashing function
def hash(
        coords: torch.Tensor,
        log2_hashmap_size: int
) -> torch.Tensor:
    """
    This function takes as input multidimensional coordinates (at most 7D) and compress them in a single
    discrete index for a hash table of size 2^log2_hashmap_size.
    """
    # A set of prime numbers is employed in order to distribute uniformly the indexes
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    # Initialize xor results with zero
    xor_result = torch.zeros_like(coords)[..., 0]
    
    # Fill each entry according to prime distribution
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    # Compute the final hash indexes
    hash_indexes = torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result

    return hash_indexes


def total_variation_loss(
        embeddings: torch.Tensor,
        min_resolution: int,
        max_resolution: int,
        level: int,
        log2_hashmap_size: int,
        n_levels: int
) -> torch.Tensor:
    """
    Total Variation Loss (TVL) penalizes sharp changes in the input embedding values,
    promoting geometric and spatial smoothness during optimization.
    """
    # Get resolution
    b = exp((log(max_resolution) - log(min_resolution)) / (n_levels - 1))
    resolution = torch.tensor(floor(min_resolution * b**level)).to(cfg_parameters["device"])

    # Cube size to apply TV loss
    min_cube_size = min_resolution - 1
    max_cube_size = 100
    cube_size = torch.floor(torch.clip(resolution / 10.0, min_cube_size, max_cube_size)).int()

    # Sample cuboid
    min_vertex = torch.randint(0, resolution - cube_size, (3,)).to(cfg_parameters["device"])
    idx = min_vertex + torch.stack([torch.arange(cube_size + 1).to(cfg_parameters["device"]) for _ in range(3)], dim=-1)
    
    # FIX: Aggiunto indexing='ij' esplicito per prevenire warning ed errori di allineamento
    cube_indices = torch.stack(torch.meshgrid(idx[:, 0], idx[:, 1], idx[:, 2], indexing='ij'), dim=-1)

    hashed_indices = hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)

    # Compute loss
    tv_x = torch.pow(cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :], 2).sum()
    tv_y = torch.pow(cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :], 2).sum()
    tv_z = torch.pow(cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :], 2).sum()
    tvl_val = ((tv_x + tv_y + tv_z) / cube_size)

    return tvl_val


def sigma_sparsity_loss(
        sigmas: torch.Tensor
) -> torch.Tensor:
    """
    Simple implementation of sparsity loss according to Cauchy.
    Points skipped by the OccupancyGrid will pass sigmas=0, yielding a 0 loss contribution.
    """
    # Using Cauchy Sparsity loss on sigma values
    sparsity_loss_val = torch.log(1.0 + 2 * sigmas**2).sum(dim=-1)

    return sparsity_loss_val


class TunableBimodalDrivingLoss(nn.Module):
    """
    Implements a flexible driving objective utilizing pseudo-labels (e.g. DINO + SAM2)
    with a continuous softness configuration parameter.
    """
    def __init__(
            self, 
            softness: float = 0.25, 
            gamma: float = 1.2, 
            w_bg: float = 4.0, 
            w_fg: float = 1.0, 
            dice_weight: float = 0.5,
            epsilon: float = 1e-5
    ):
        super(TunableBimodalDrivingLoss, self).__init__()
        self.softness = max(0.0, min(1.0, softness))
        self.gamma = gamma
        self.w_bg = w_bg
        self.w_fg = w_fg
        self.dice_weight = dice_weight
        self.epsilon = epsilon

    def forward(
            self, 
            pred_mask: torch.Tensor, 
            pseudo_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the relaxed optimization cost over dynamically blended target references.
        """
        pred = pred_mask.view(-1)
        target = pseudo_target.view(-1)

        # --- CONTINUOUS TARGET SOFTENING TRANSFORMATION ---
        target_tuned = self.softness * 0.5 + (1.0 - self.softness) * torch.pow(target, self.gamma)

        # 1. Asymmetric Penalization Component
        bg_loss = torch.mean((1.0 - target_tuned) * torch.square(pred))
        fg_loss = torch.mean(target_tuned * torch.abs(pred - target_tuned))
        asymmetric_term = (self.w_bg * bg_loss) + (self.w_fg * fg_loss)

        # 2. Soft-Dice Layout Loss Component
        intersection = torch.sum(pred * target_tuned)
        union = torch.sum(torch.square(pred)) + torch.sum(torch.square(target_tuned))
        dice_term = 1.0 - (2.0 * intersection + self.epsilon) / (union + self.epsilon)

        total_driving_loss = asymmetric_term + (self.dice_weight * dice_term)
        return total_driving_loss


class BCEDiceLoss(nn.Module):
    """
    Standard mixed loss between BinaryCrossEntropy and Dice losses for segmentation tasks.
    """
    def __init__(self, alpha: float, beta: float):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (inputs.sum() + targets.sum() + 1e-6)

        bcde_dice_loss = self.alpha * bce_loss + self.beta * dice_loss
        return bcde_dice_loss