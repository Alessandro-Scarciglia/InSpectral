# Import modules
import torch
import torch.optim.lr_scheduler as lr_scheduler

# Import custom modules
from losses import total_variation_loss, BCEDiceLoss
from metrics import *

# Parameters
from config_parameters import *


class Trainer:
    """
    This class implements a trainer agent. It performs the training and the validation of a single iteration (batch).
    Its methods are called several times during the training and modify inplace the learning parameters.

    Attributes:
    ----------
    model: every PyTorch-supported model format
        it is the model to be optimized.
    lr: float
        the learning rate of the optimization.
    betas: tuple[float, float]
        beta1 and beta2, usually (0.9, 0.99) are parameters which regulate the mean gradient and the root mean square
        of gradients (i.e. the regulate first and second moments of the gradient). Beta1 takes into account the old gradients,
        whereas beta2 stabilizes the scale of the gradient.
        All in all, both beta1 and beta2 manage the memory of the previous gradients: the former for the mean, the latter for
        the variance. The number of steps are 1/(1-beta).
    eps: float
        it is a numerical trick to improve numerical stability when normalizing the gradients. Typical value is 1e-8.
    tv_loss_weight: float
        the weight of the total variation loss in the optimization.
    sparsity_loss_weight: float
        the weight of the sparsity loss in the optimization.
    decay_rate: float
        the rate of decay of the leraning rate, in a manner like: lr * (decay_rate)^n where n is the 
        number of of the current epoch.
    decay_steps: int
        it is the number of epoch when the decay shall start, thus: lr * (decay_rate)^(n-decay_steps) when n-decay_steps >= 0. 
    """
    def __init__(
            self,
            model,
            lr: float,
            betas: tuple,
            eps: float,
            tv_loss_weight: float,
            sparsity_loss_weight: float,
            decay_rate: float,
            decay_steps: int
    ):
        
        # Attributes
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.tot_var_w = tv_loss_weight
        self.sparsity_loss_w = sparsity_loss_weight
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.params = hash_parameters
        self.bcedice = BCEDiceLoss()

        # Instantiate the optimizer
        # TODO: Investigate on the use of single vs. multiple groups.
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.model.nerf.parameters(), 'lr': self.lr},
                {'params': self.model.embedder.parameters(), 'lr': self.lr},
            ],
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
            sundir: torch.Tensor,
            labels: torch.Tensor,
            epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function implements the training step of a single batch.

        Parameters:
        ----------
        rays: torch.Tensor[float]
            a tensor containing input rays origins and directions.
        sundir: torch.Tensor[float]
            a tensor containing the direction of the light source.
        labels: torch.Tensor[float]
            a tensor  containing the truth for output channels of pixels.
        epoch: int
            current epoch of optimization.

        Returns:
        -------
        loss: torch.Tensor[float]
            it is the cumulative loss, sum of photometric loss, sparsity loss, BCEDice and TVL.
        loss_photom: torch.Tensor[float]
            it is the photometric loss (MSE).
        loss_tv: torch.Tensor[float]
            it is the Total Variation Loss on the embeddings.
        loss_sparsity: torch.Tensor[float]
            it is the sparsity loss on the object 3D geometry.
        loss_bcedice: torch.Tensor[float]
            it is the BCEDice loss on the object segmentation mask estimate. 
        """
        
        # Move labels and sundir to device
        labels_rgb = labels[:, 0].unsqueeze(-1).to(self.model.device)
        labels_mask = labels[:, 1].unsqueeze(-1).to(self.model.device)

        # Forward pass
        chs_map, _, loss_sparsity, mask = self.model(rays, sundir)

        # Zeroing the gradient
        self.optimizer.zero_grad()

        # Compute Photometric Loss on pixel estimate (MSE)
        loss_photom = compute_mse(chs_map, labels_rgb)

        # Compute Total Variation Loss (TVL) on hash embeddings
        loss_tv = sum(
            total_variation_loss(
                self.model.embedder.embeddings[i],
                hash_parameters["low_res"],
                hash_parameters["high_res"],
                i, hash_parameters["log2_hashmap_size"],
                hash_parameters["n_levels"]) for i in range(hash_parameters["n_levels"]
            )
        )

        # Compute BCEDice loss on segmentation masks
        loss_bce_dice = self.bcedice(mask, labels_mask)

        # Combine the computed losses. A basic loss function is given by the weighted sum of the 
        # photometric loss with the sparsity loss. Optionally, other losses can be added
        loss = loss_photom + \
            training_parameters["sparsity_loss_weight"] * loss_sparsity
        
        # For some epochs, add the total variation loss
        if epoch <= training_parameters["stop_tv_epoch"]:
            loss += training_parameters["tv_loss_weight"] * loss_tv

        # In the last epochs, add the BCE-Dice loss
        if epoch >= training_parameters["start_seg_epoch"]:
            loss += training_parameters["bce_dice_loss_weight"] * loss_bce_dice

        # Perform backpropagation
        loss.backward()
        self.optimizer.step()

        return loss, loss_photom, loss_tv, loss_sparsity, loss_bce_dice

    def valid_one_batch(
            self,
            rays: torch.Tensor,
            labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function implements the validation phase on a single batch of the Validation Set.

        Parameters:
        ----------
        rays: torch.Tensor[float]
            a tensor containing input rays origins and directions.
        labels: torch.Tensor[float]
            a tensor  containing the truth for output channels of pixels.

        Returns:
        -------
        loss: torch.Tensor[float]
            it is the cumulative loss, sum of photometric loss, sparsity loss, BCEDice and TVL.
        loss_photom: torch.Tensor[float]
            it is the photometric loss (MSE).
        loss_sparsity: torch.Tensor[float]
            it is the sparsity loss on the object 3D geometry.
        psnr: torch.Tensor[float]
            the Power Signal-to-Noise Ratio between the rendered frame and the captured one.
        """
        # Move labels to device
        labels = labels.to(self.model.device)

        # Forward pass
        chs_map, _, loss_sparsity, _ = self.model(rays)

        # Compute photometric loss on pixel estimate
        loss_photom = compute_mse(chs_map, labels)
        
        # Combinate losses
        # TODO: Why here the loss does not include TVL and BCEDice?
        loss = loss_photom + \
            training_parameters["sparsity_loss_weight"] * loss_sparsity
        
        # Compute PSNR
        psnr = compute_psnr(
            img1=chs_map,
            img2=labels
        )
        
        return loss, loss_photom, loss_sparsity, psnr