# Import modules
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
        
        # Attributes
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
        self.bcedice = BCEDiceLoss(
            alpha=alpha,
            beta=beta
        )

        # Instantiate the optimizer with grouped parameters
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
        
        # Create a scheduler
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
        
        # Move labels to device
        labels_rgb = labels[:, :3].to(self.model.device)
        labels_mask = labels[:, -1].unsqueeze(-1).to(self.model.device)

        # --- 1. CALCOLO STEP GLOBALE LINEARE ---
        current_step = (epoch * dl_len) + n_iter

        # --- 2. DINAMICA DEL PESO DELLA SPARSITY LOSS (WARM-UP) ---
        sparsity_warmup_end = max_steps * 0.15  # Il warm-up dura per il primo 15% del training
        if current_step < sparsity_warmup_end:
            sparsity_factor = current_step / sparsity_warmup_end
        else:
            sparsity_factor = 1.0
        current_sparsity_weight = training_parameters["sparsity_loss_weight"] * sparsity_factor

        # Forward pass (Passiamo i parametri temporali corretti per il Coarse-to-Fine dell'hash)
        chs_map, _, loss_sparsity, mask = self.model(rays, app_idx, epoch, n_iter, max_steps, dl_len)

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
                hash_parameters["n_levels"]) for i in range(hash_parameters["n_levels"])
        )

        # Compute the sum of cosine similarities in adjacent embeddings
        loss_app_sim = torch.mean(
            (self.model.app_embedder.embeddings.weight[2:] + \
            - 2 * self.model.app_embedder.embeddings.weight[1:-1] + \
            self.model.app_embedder.embeddings.weight[:-2]) ** 2
        )

        # Compute BCEDice loss on segmentation masks
        loss_bce_dice = self.bcedice(mask, labels_mask)

        # Combine the computed losses using the dynamic weight scaled by the warm-up
        loss = loss_photom + \
            current_sparsity_weight * loss_sparsity + \
            training_parameters["similarity_loss_weight"] * loss_app_sim
        
        # For some epochs, add the total variation loss
        if epoch <= training_parameters["stop_tv_epoch"]:
            loss += training_parameters["tv_loss_weight"] * loss_tv

        # In the last epochs, add the BCE-Dice loss
        if epoch >= training_parameters["start_seg_epoch"]:
            loss += training_parameters["bce_dice_loss_weight"] * loss_bce_dice

        # Perform backpropagation
        loss.backward()
        self.optimizer.step()

        return loss, loss_photom, loss_tv, loss_sparsity, loss_bce_dice, loss_app_sim

    def valid_one_batch(
            self,
            rays: torch.Tensor,
            labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Move labels to device
        labels_rgb = labels[:, :3].to(self.model.device)

        # --- CORREZIONE VALIDAZIONE ---
        # Forniamo argomenti di fallback per i vincoli temporali (None indica all'embedder di usare 100% dei livelli)
        # Passiamo 0 come appearance index di default o una media se previsto dal modello
        chs_map, _, loss_sparsity, _ = self.model(rays, torch.tensor([0], device=self.model.device), None, None, None, None)

        # Compute photometric loss on pixel estimate
        loss_photom = compute_mse(chs_map, labels_rgb)
        
        # Combine losses
        loss = loss_photom + training_parameters["sparsity_loss_weight"] * loss_sparsity
        
        # Compute PSNR
        psnr = compute_psnr(
            img1=chs_map,
            img2=labels_rgb
        )
        
        return loss, loss_photom, loss_sparsity, psnr