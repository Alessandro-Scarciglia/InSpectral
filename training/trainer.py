"""
This module handles the training and validation loops for a joint NeRF and Pose Optimization model.
It implements grouped parameter optimization, a constant sparsity loss setup, 
and temporal-constrained multi-task training (Photometric, Total Variation, Tunable Driving, and Appearance Similarity).
"""

# Import standard modules
import torch
import torch.optim.lr_scheduler as lr_scheduler

# Import custom modules
from model.losses import total_variation_loss, TunableBimodalDrivingLoss, BCEDiceLoss
from model.metrics import *


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
            photo_loss_weight: float,
            tv_loss_weight: float,
            sparsity_loss_weight: float,
            similarity_loss_weight: float,
            decay_rate: float,
            decay_steps: int,
            stop_tv_epoch: int,         
            start_seg_epoch: int,        
            driving_loss_weight: float,  
            softness: float = 0.25,  
            gamma: float = 1.2,
            enable_camera_refinement: bool = False,
            camera_warmup_epochs: int = 0,
            camera_update_interval: int = 1,
            camera_only: bool = False,
            camera_pose_reg_weight: float = 0.0,
            mixed_precision: bool = False,
            distortion_loss_weight: float = 0.0
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
        
        # Register loss weights and temporal thresholds as instance attributes
        self.photo_loss_weight = photo_loss_weight
        self.tot_var_w = tv_loss_weight
        self.sparsity_loss_w = sparsity_loss_weight
        self.similarity_loss_w = similarity_loss_weight
        self.driving_loss_w = driving_loss_weight
        self.stop_tv_epoch = stop_tv_epoch
        self.start_seg_epoch = start_seg_epoch
        self.enable_camera_refinement = bool(enable_camera_refinement)
        self.camera_warmup_epochs = max(0, int(camera_warmup_epochs))
        self.camera_update_interval = max(1, int(camera_update_interval))
        self.camera_only = bool(camera_only)
        self.camera_pose_reg_weight = max(0.0, float(camera_pose_reg_weight))
        self.distortion_loss_w = max(0.0, float(distortion_loss_weight))
        self.mixed_precision = bool(mixed_precision and torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler(enabled=self.mixed_precision)
        
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
        # Extract properties directly from the embedder object to eliminate external config dependencies
        self.hash_params = {
            "low_res": getattr(model.embedder, "low_resolution", getattr(model.embedder, "low_res", 16)),
            "high_res": getattr(model.embedder, "high_resolution", getattr(model.embedder, "high_res", 512)),
            "log2_hashmap_size": getattr(model.embedder, "log2_hashmap_size", 19),
            "n_levels": getattr(model.embedder, "n_levels", 16)
        }
        
        # New Tunable driving loss replacing the strict BCE+Dice setup
        # self.driving_loss = TunableBimodalDrivingLoss(
        #     softness=softness, 
        #     gamma=gamma,
        #     w_bg=4.0, 
        #     w_fg=1.0
        # )

        self.driving_loss = BCEDiceLoss(
            alpha=0.5,
            beta=0.5
        )

        #self.driving_loss = TunableBimodalDrivingLoss()

        # Setup optimizer grouping
        self.optimizer = torch.optim.AdamW(
            [
                {'params': filter(lambda p: p.requires_grad, self.model.nerf.parameters()), 'lr': self.lr},
                {'params': filter(lambda p: p.requires_grad, self.model.embedder.parameters()), 'lr': self.emb_lr},
                {'params': filter(lambda p: p.requires_grad, self.model.app_embedder.parameters()), 'lr': self.app_lr},
                {'params': filter(lambda p: p.requires_grad, self.res_model.parameters()), 'lr': self.res_lr}
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

    def _set_optimization_phase(self, epoch: int, n_iter: int) -> bool:
        """Freeze groups for alternating camera-only updates.

        Returns whether this batch updates the pose table.  Refinement is opt-in
        and the default path leaves all parameters trainable as before.
        """
        active = (self.enable_camera_refinement and
                  epoch >= self.camera_warmup_epochs and
                  (n_iter % self.camera_update_interval == 0))
        camera_only = active and self.camera_only
        for parameter in self.model.parameters():
            parameter.requires_grad_(not camera_only)
        for parameter in self.res_model.parameters():
            parameter.requires_grad_(active)
        return active

    def train_one_batch(
            self,
            rays: torch.Tensor,
            labels: torch.Tensor,
            driving_mask: torch.Tensor,  
            app_idx: torch.Tensor,
            epoch: int,
            n_iter: int,
            max_steps: int,
            dl_len: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Executes a single optimization iteration over a pixel/ray training batch.
        """
        # labels contiene esattamente 3 canali (RGB) passati dallo script principale
        labels_rgb = labels[:, :3].to(self.model.device)   
        labels_mask = driving_mask.to(self.model.device)

        pose_active = self._set_optimization_phase(epoch, n_iter)
        self.optimizer.zero_grad()
        amp_dtype = torch.float16
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=self.mixed_precision,
        ):
            # Forward pass and loss construction share one autocast region.
            chs_map, _, loss_sparsity, mask = self.model(
                rays, app_idx, epoch, n_iter, max_steps, dl_len
            )

            # Core components that are always active in the computational graph.
            loss_photom = compute_mse(chs_map, labels_rgb)

            loss_app_sim = torch.mean(
                (self.model.app_embedder.embeddings.weight[2:]
                 - 2 * self.model.app_embedder.embeddings.weight[1:-1]
                 + self.model.app_embedder.embeddings.weight[:-2]) ** 2
            )

            # Initialize the total loss with fixed active components.
            loss = (
                self.photo_loss_weight * loss_photom
                + self.sparsity_loss_w * loss_sparsity
                + self.similarity_loss_w * loss_app_sim
            )
            loss_distortion = self.model.integrator.last_distortion_loss.float()
            if self.distortion_loss_w:
                loss = loss + self.distortion_loss_w * loss_distortion
            if pose_active and self.camera_pose_reg_weight:
                pose_reg = self.res_model.embeddings.weight.square().mean()
                loss = loss + self.camera_pose_reg_weight * pose_reg
        
        # --- TOTAL VARIATION LOSS (HYBRID / SHADOW METRIC) ---
        if epoch <= self.stop_tv_epoch:
            
            # Active computation inside the gradient graph
            loss_tv = sum(
                total_variation_loss(
                    self.model.embedder.embeddings[i],
                    self.hash_params["low_res"], 
                    self.hash_params["high_res"],
                    i, self.hash_params["log2_hashmap_size"],
                    self.hash_params["n_levels"]
                ) for i in range(self.hash_params["n_levels"])
            )
            loss = loss + (self.tot_var_w * loss_tv)
        else:
            # Calculated without gradient after stop_tv_epoch to maintain tracking on Tensorboard
            with torch.no_grad():
                loss_tv = sum(
                    total_variation_loss(
                        self.model.embedder.embeddings[i],
                        self.hash_params["low_res"], 
                        self.hash_params["high_res"],
                        i, self.hash_params["log2_hashmap_size"],
                        self.hash_params["n_levels"]
                    ) for i in range(self.hash_params["n_levels"])
                )

        # --- DRIVING LOSS (HYBRID / SHADOW METRIC) ---
        if epoch >= self.start_seg_epoch:
            # Active computation inside the gradient graph
            loss_driving = self.driving_loss(mask, labels_mask)
            loss = loss + (self.driving_loss_w * loss_driving)
        else:
            # Calculated without gradient before start_seg_epoch for diagnostic purposes
            with torch.no_grad():
                loss_driving = self.driving_loss(mask, labels_mask)

        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss, loss_photom, loss_tv, loss_sparsity, loss_driving, loss_app_sim, loss_distortion