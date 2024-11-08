'''
This module implements the training agent.
This function is callable and produces one-iteration training step.
'''

# Import modules
import torch
import torch.optim.lr_scheduler as lr_scheduler

# Import custom modules
from losses import total_variation_loss
from metrics import *

# Parameters
from parameters import *


# Trainer agent class implementation
class Trainer:
    def __init__(
            self,
            model,
            lr: float = 0.001,
            betas: tuple = (0.9, 0.999),
            eps: float = 1e-8,
            tot_var_weight: float = 1e-6,
            sparsity_loss_weight: float = 1e-10,
            decay_rate: float = 1e-1,
            decay_steps: int = 1000
    ):
        
        # Attributes
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.tot_var_w = tot_var_weight
        self.sparsity_loss_w = sparsity_loss_weight
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.params = hash_parameters

        # Lambdas
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)

        # Instantiate the optimizer
        self.optimizer = torch.optim.Adam(
            [{'params': self.model.nerf.parameters(), 'lr': self.lr},
            {'params': self.model.embedder.parameters(), 'lr': self.lr}],
            betas=self.betas,
            eps=self.eps
        )
        
        # Create a scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             step_size=decay_steps,
                                             gamma=decay_rate,
                                             verbose=training_parameters["verbose"])

    def train_one_batch(
            self,
            rays: torch.Tensor,
            labels: torch.Tensor,
            epoch: int
    ):
        
        # Move labels to device
        labels = labels.to(self.model.device)

        # Forward pass
        chs_map, _, loss_sparsity = self.model(rays)

        # Zero the gradient
        self.optimizer.zero_grad()

        # Compute photometric loss on pixel estimate
        loss_photom = self.img2mse(chs_map, labels)

        # Compute Total Variation Loss on position embeddings
        loss_tv = sum(
            total_variation_loss(
                self.model.embedder.embeddings[i],
                hash_parameters["low_res"],
                hash_parameters["high_res"],
                i, hash_parameters["log2_hashmap_size"],
                hash_parameters["n_levels"]) for i in range(hash_parameters["n_levels"]
            )
        )

        # Combinate losses
        loss = loss_photom + \
            training_parameters["sparsity_loss_weight"] * loss_sparsity
        
        # For some epochs, add the total variation loss
        if epoch <= training_parameters["stop_tv_epoch"]:
            loss += training_parameters["tv_loss_weight"] * loss_tv

        # Backprop
        loss.backward()
        self.optimizer.step()

        return loss, loss_photom, loss_tv, loss_sparsity


    def valid_one_batch(
            self,
            rays: torch.Tensor,
            labels: torch.Tensor
    ):
        
        # Move labels to device
        labels = labels.to(self.model.device)

        # Forward pass
        chs_map, _, loss_sparsity = self.model(rays)

        # Compute photometric loss on pixel estimate
        loss_photom = self.img2mse(chs_map, labels)
        
        # Combinate losses
        loss = loss_photom + \
            training_parameters["sparsity_loss_weight"] * loss_sparsity
        
        # Compute PSNR
        psnr = compute_psnr(
            img1=chs_map,
            img2=labels
        )
        
        return loss, loss_photom, loss_sparsity, psnr