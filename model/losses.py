"""
This module contains the custom loss functions used during the training of the NeRF model.
Includes Hashing, Total Variation, Cauchy Sparsity, and Tunable Bimodal Driving Loss.
"""

# Import modules
from math import exp, log, floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config_parameters import *


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
    # tv_x = torch.pow(cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :], 2).sum()
    # tv_y = torch.pow(cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :], 2).sum()
    # tv_z = torch.pow(cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :], 2).sum()
    tv_x = torch.abs(cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :]).sum()
    tv_y = torch.abs(cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :]).sum()
    tv_z = torch.abs(cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :]).sum()
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
    Custom Loss Module for driving NeRF geometry using noisy/estimated pseudo-labels 
    (e.g., outputs from vision foundation models like DINOv2 or SAM2).

    Key Features:
    -------------
    1. Confidence Adjustment (Gamma Correction): Modulates target sharpness/uncertainty.
    2. Controlled Label Smoothing: Softens extreme predictions along noisy object boundaries.
    3. Asymmetric Penalization (L2 vs L1): Uses a strict L2 penalty for background regions 
       to aggressively suppress floating artifacts (floaters), and a more tolerant L1 
       penalty for foreground regions to retain fine details.
    4. Soft-Dice Loss: Enforces global structural layout consistency across viewpoints.
    """
    def __init__(
            self, 
            softness: float = 0.05,  # Keep low (e.g., 0.02 - 0.05) to avoid foggy background baselines
            gamma: float = 1.0,      # Controls non-linear target scaling (gamma > 1 sharpens, < 1 softens)
            w_bg: float = 4.0,       # High background weight to aggressively clear out artifacts
            w_fg: float = 1.0,       # Foreground weight for object geometry reconstruction
            dice_weight: float = 0.5, # Balance coefficient for global layout structural loss
            epsilon: float = 1e-5    # Small constant to prevent division by zero in Dice loss
    ):
        """
        Initializes hyperparameters for target adjustment and hybrid loss terms.

        Args:
            softness (float): Clamped amount of target interpolation towards 0.5. Range: [0.0, 0.2].
            gamma (float): Exponent power applied to target probabilities to reshape confidence.
            w_bg (float): Multiplier weight for background region penalty.
            w_fg (float): Multiplier weight for foreground region penalty.
            dice_weight (float): Multiplier weight for the Soft-Dice loss component.
            epsilon (float): Numerical stability constant.
        """
        super(TunableBimodalDrivingLoss, self).__init__()
        
        # Clamp softness between 0.0 and 0.2 to prevent excessive baseline drift
        self.softness = max(0.0, min(0.2, softness))
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
        Computes the combined Asymmetric L1/L2 and Soft-Dice objective.

        Args:
            pred_mask (torch.Tensor): Model predictions (e.g., rendered ray opacities or soft 
                                      segmentation logits mapped to [0, 1]).
            pseudo_target (torch.Tensor): Estimated pseudo-labels from SAM2/DINO in range [0, 1].

        Returns:
            torch.Tensor: Scalar loss tensor combining asymmetric penalties and soft-dice structural loss.
        """
        # Flatten spatial/ray dimensions into 1D vectors for efficient vectorized operations
        pred = pred_mask.view(-1)
        target = pseudo_target.view(-1)

        # -------------------------------------------------------------------------
        # 1. Target Confidence Reshaping (Gamma Correction)
        # -------------------------------------------------------------------------
        # Gamma scaling selectively shifts confidence curves.
        # Gamma > 1.0 suppresses uncertain background noise (pushes low probabilities closer to 0).
        # Gamma < 1.0 boosts low-confidence foreground regions.
        if self.gamma != 1.0:
            target = torch.pow(target, self.gamma)

        # -------------------------------------------------------------------------
        # 2. Controlled Label Smoothing
        # -------------------------------------------------------------------------
        # Contracts target probabilities towards 0.5 according to `self.softness`.
        # Softens extreme binary targets (0 -> softness/2, 1 -> 1 - softness/2)
        # to prevent gradient explosion on view-inconsistent mask boundaries.
        if self.softness > 0.0:
            target_tuned = target * (1.0 - self.softness) + 0.5 * self.softness
        else:
            target_tuned = target

        # -------------------------------------------------------------------------
        # 3. Asymmetric Penalization Component (L2 Background / L1 Foreground)
        # -------------------------------------------------------------------------
        # Background Loss (L2):
        # Weighted by (1.0 - target_tuned). Uses quadratic penalty (pred^2) to exert strong
        # downward pressure on low-density predictions, clearing out floating 3D density artifacts.
        bg_loss = torch.mean((1.0 - target_tuned) * torch.square(pred))

        # Foreground Loss (L1):
        # Weighted by target_tuned. Uses absolute difference |pred - target| to provide linear,
        # stable gradients that tolerate minor view-inconsistencies along fine object boundaries.
        fg_loss = torch.mean(target_tuned * torch.abs(pred - target_tuned))
        
        # Combine weighted background and foreground components
        asymmetric_term = (self.w_bg * bg_loss) + (self.w_fg * fg_loss)

        # -------------------------------------------------------------------------
        # 4. Soft-Dice Layout Loss Component
        # -------------------------------------------------------------------------
        # Computes overlap over continuous predictions and targets. Ensures global structural
        # consistency and prevents partial object collapse across views.
        intersection = torch.sum(pred * target_tuned)
        union = torch.sum(torch.square(pred)) + torch.sum(torch.square(target_tuned))
        dice_term = 1.0 - (2.0 * intersection + self.epsilon) / (union + self.epsilon)

        # -------------------------------------------------------------------------
        # 5. Total Combined Loss Output
        # -------------------------------------------------------------------------
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
        # Renderer masks are already probabilities in [0, 1], not logits.
        # Clamping only protects log(0) while retaining stable BCE gradients.
        self.bce = nn.BCELoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.clamp(1e-6, 1.0 - 1e-6)
        bce_loss = self.bce(inputs, targets.clamp(0.0, 1.0))

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (inputs.sum() + targets.sum() + 1e-6)

        bcde_dice_loss = self.alpha * bce_loss + self.beta * dice_loss
        return bcde_dice_loss