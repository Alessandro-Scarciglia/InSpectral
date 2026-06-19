"""
This module handles the training and validation loops for a joint NeRF and Pose Optimization model.
It implements grouped parameter optimization, dynamic sparsity loss warm-up schedules, 
and temporal-constrained multi-task training (Photometric, Total Variation, BCE-Dice, and Appearance Similarity).
"""

# Import standard modules
import torch
import torch.optim.lr_scheduler as lr_scheduler

# Import custom modules
from model.losses import total_variation_loss, BCEDiceLoss
from model.metrics import *

# Parameters
from config_parameters import *


class Trainer:
    def __init__(
            self,
            model,
            res_model,
            lr: float,
            lr_res: float,
            lr_emb: float,
            lr_app: float,
            betas: tuple,
            eps: float,
            tv_loss_weight: float,
            sparsity_loss_weight: float,
            decay_rate: float,
            decay_steps: int,
            alpha: float,
            beta: float
    ):
        """
        Initializes the optimization agent, groups architectural parameters, 
        and sets up scheduling policies.
        """
        # Attributes assignment
        self.model = model
        self.res_model = res_model
        self.lr = lr
        self.res_lr = lr_res
        self.emb_lr = lr_emb
        self.app_lr = lr_app
        self.betas = betas
        self.eps = eps
        self.tot_var_w = tv_loss_weight
        self.sparsity_loss_w = sparsity_loss_weight
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.params = hash_parameters
        
        # Binary Cross Entropy + Dice loss helper for semantic masking
        self.bcedice = BCEDiceLoss(alpha=alpha, beta=beta)

        # Instantiate the per-group learning rate learning optimizer
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.model.nerf.parameters(), 'lr': self.lr},
                {'params': self.model.embedder.parameters(), 'lr': self.emb_lr},
                {'params': self.model.app_embedder.parameters(), 'lr': self.app_lr},
                {'params': self.res_model.parameters(), 'lr': self.res_lr}
            ],
            betas=self.betas,
            eps=self.eps
        )
        
        # Setup learning rate scheduler policy
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=decay_steps,
            gamma=decay_rate,
        )

    def train_one_batch(
            self,
            rays: torch.Tensor,
            labels: torch.Tensor,
            app_idx: torch.Tensor,
            epoch: int,
            n_iter: int,
            max_steps: int,
            dl_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes a single optimization iteration over a pixel/ray training batch.
        """
        # Move target supervision batches onto the correct device
        labels_rgb = labels[:, :3].to(self.model.device)
        labels_mask = labels[:, -1].unsqueeze(-1).to(self.model.device)

        # -------------------------------------------------------------------------------
        # 1. GLOBAL LINEAR STEP COMPUTATION
        # -------------------------------------------------------------------------------
        current_step = (epoch * dl_len) + n_iter

        # -------------------------------------------------------------------------------
        # 2. SPARSITY LOSS WEIGHT WARM-UP DYNAMICS
        # -------------------------------------------------------------------------------
        # Linearly ramp up sparsity weight during the first 15% of training steps
        sparsity_warmup_end = max_steps * 0.15  
        if current_step < sparsity_warmup_end:
            sparsity_factor = current_step / sparsity_warmup_end
        else:
            sparsity_factor = 1.0
        current_sparsity_weight = training_parameters["sparsity_loss_weight"] * sparsity_factor

        # -------------------------------------------------------------------------------
        # 3. FORWARD INFRASTRUCTURE PASS
        # -------------------------------------------------------------------------------
        # Feed-forward passing structural anchors for grid level progressive training
        chs_map, _, loss_sparsity, mask = self.model(rays, app_idx, epoch, n_iter, max_steps, dl_len)

        # Clear active gradients before backward calculations
        self.optimizer.zero_grad()

        # Compute standard Mean Squared Error (MSE) Photometric loss
        loss_photom = compute_mse(chs_map, labels_rgb)

        # Compute multi-level Total Variation Loss (TVL) over grid features
        loss_tv = sum(
            total_variation_loss(
                self.model.embedder.embeddings[i],
                hash_parameters["low_res"],
                hash_parameters["high_res"],
                i, hash_parameters["log2_hashmap_size"],
                hash_parameters["n_levels"]) for i in range(hash_parameters["n_levels"])
        )

        # Evaluate smoothness constraints by penalizing local appearance changes (second derivative)
        loss_app_sim = torch.mean(
            (self.model.app_embedder.embeddings.weight[2:] + \
            - 2 * self.model.app_embedder.embeddings.weight[1:-1] + \
            self.model.app_embedder.embeddings.weight[:-2]) ** 2
        )

        # Evaluate structural cross-entropy on object borders
        loss_bce_dice = self.bcedice(mask, labels_mask)

        # -------------------------------------------------------------------------------
        # 4. CONDITIONAL MULTI-TASK LOSS AGGREGATION
        # -------------------------------------------------------------------------------
        # Assemble standard core objectives
        loss = loss_photom + \
            current_sparsity_weight * loss_sparsity + \
            training_parameters["similarity_loss_weight"] * loss_app_sim
        
        # Apply Total Variation (TV) localization up to the threshold epoch limit
        if epoch <= training_parameters["stop_tv_epoch"]:
            loss += training_parameters["tv_loss_weight"] * loss_tv

        # Inject semantic mask segmentation objectives in the final training phase
        if epoch >= training_parameters["start_seg_epoch"]:
            loss += training_parameters["bce_dice_loss_weight"] * loss_bce_dice

        # -------------------------------------------------------------------------------
        # 5. BACKWARD MINIMIZATION STEP
        # -------------------------------------------------------------------------------
        loss.backward()
        self.optimizer.step()

        return loss, loss_photom, loss_tv, loss_sparsity, loss_bce_dice, loss_app_sim

    def valid_one_batch(
            self,
            rays: torch.Tensor,
            labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes an isolated evaluation pass without modifying structural weight gradients.
        """
        # Move target arrays to device
        labels_rgb = labels[:, :3].to(self.model.device)

        # -------------------------------------------------------------------------------
        # VALIDATION BOUNDARY FALLBACK
        # -------------------------------------------------------------------------------
        # Passing None to time variables ensures that all features levels are fully active 
        # during inference. Fallback to index 0 for the appearance embedding.
        fallback_idx = torch.tensor([0], device=self.model.device)
        chs_map, _, loss_sparsity, _ = self.model(rays, fallback_idx, None, None, None, None)

        # Compute metrics on pixel estimations
        loss_photom = compute_mse(chs_map, labels_rgb)
        loss = training_parameters["photometric_loss_weight"] * loss_photom + training_parameters["sparsity_loss_weight"] * loss_sparsity
        
        # Calculate Peak Signal-to-Noise Ratio (PSNR)
        psnr = compute_psnr(img1=chs_map, img2=labels_rgb)
        
        return loss, loss_photom, loss_sparsity, psnr