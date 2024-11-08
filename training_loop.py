'''
This file is meant to build the training process of the rendering model.
'''

# Import custom modules
from rays_generator import RaysGenerator
from rendering import NeuralRenderer
from dataset import InspectralData
from trainer import Trainer
from metrics import *

# Parameters
from parameters import *

# Import standard modules
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import json
from datetime import datetime
import os
import time

# Set seeds for test repeatibility
torch.manual_seed(1234)


# Training function
def main(folder_name: str):
    
    # Load calibration matrix
    with open("calibration/calibration.json", "r") as fopen:
        K = json.load(fopen)["mtx"]
        K = np.array(K).reshape(3, 3)
        K[:2, :3] /= (1024 / cfg_parameters["resolution"])

    # Initialize rays generator
    ray_gen = RaysGenerator(
        **rays_parameters,
        K=K
    )

    # Instantiate the model
    model = NeuralRenderer(
        **sampler_parameters,
        **sampler_parameters,
        **posenc_parameters,
        **sh_parameters,
        **nerf_parameters,
        device="cuda"
    )

    # Load training and validation datasets
    training_dataset = InspectralData(**dataset_parameters, mode='training')
    validation_dataset = InspectralData(**dataset_parameters, mode='validation')

    # Create a dataloader for training and validation datasets
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=training_parameters["training_batch"],
        shuffle=True,
        num_workers=2
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=training_parameters["validation_batch"],
        shuffle=True,
        num_workers=2
    )

    # Define the Trainer class
    trainer_agent = Trainer(
        model=model,
        lr=training_parameters["lr"],
        betas=training_parameters["betas"],
        eps=training_parameters["eps"],
        tv_loss_weight=training_parameters["tv_loss_weight"],
        sparsity_loss_weight=training_parameters["sparsity_loss_weight"],
        decay_rate=training_parameters["decay_rate"],
        decay_steps=training_parameters["decay_steps"]
    )

    # Start timing
    t_start = time.time()
    
    # Training loop
    for epoch in range(training_parameters["epochs"]):

        # Set model in training mode
        model.train()
        
        # Iterate through rays samples of the training set
        for n_iter, ray_batch in enumerate(training_dataloader):

            # Train one batch
            losses = trainer_agent.train_one_batch(
                rays=ray_batch[:, :7],
                labels=ray_batch[:, 7:],
                epoch=epoch
            )

            # Split losses in tot loss and each component
            loss, loss_photom, loss_tv, loss_sparsity = losses
                            
            # Display loss values
            if training_parameters["verbose"]:
                print(f"Epoch: {epoch} | # Iter: {n_iter} | Elapsed time (s): {(time.time()-t_start):.3f} | "
                      f"Photometric Loss: {loss_photom.item():.5f} | TV Loss: {loss_tv.item():.5f} | Sparsity Loss: {loss_sparsity.item():.5f} | "
                      f"Tot Loss: {loss.item():.5f}")

        # Test model on validation set
        with torch.no_grad():

            # Create metrics buffer
            model.eval()
            valid_psnr_set = []
            
            # Loop through the validation set
            for m_iter, ray_batch_val in enumerate(validation_dataloader):

                # Process a batch and compute PSNR
                out = trainer_agent.valid_one_batch(
                    rays=ray_batch_val[:, :7],
                    labels=ray_batch_val[:, 7:],
                )

                # Split validation output
                valid_loss, valid_loss_photom, valid_loss_sparsity, valid_psnr = out
                valid_psnr_set.append(valid_psnr)

                # Display results
                if training_parameters["verbose"]:
                    print(f"\tValid: {epoch} | # Iter: {m_iter} | Elapsed time (s): {(time.time()-t_start):.3f} | "
                      f"Photometric Loss: {valid_loss_photom.item():.5f} | Sparsity Loss: {valid_loss_sparsity.item():.5f} | "
                      f"Tot Loss: {valid_loss.item():.5f}")

            # Store checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.trainer.optimizer.state_dict(),
                "psnr": torch.mean(valid_psnr_set).item()
            }
            epoch_folder = os.path.join(folder_name, f"epoch_{epoch}")
            os.mkdir(epoch_folder)
            torch.save(checkpoint, os.path.join(epoch_folder, "chkpt.pt"))
            
            # Print metrics
            if training_parameters["verbose"]:
                print(f"Validation after epoch {epoch} | PSNR: {torch.mean(valid_psnr_set).item():.3f}\n\n")

            # Test trajectories
            for trajectory in training_parameters["test_trajectories"]:

                # Create trajectory folder
                traj_name = trajectory.split(".")[0].split("/")[-1]
                traj_folder = os.path.join(epoch_folder, traj_name)
                os.mkdir(traj_folder)
                
                # Create frames and depths folders
                frames_folder = os.path.join(traj_folder, "frames")
                depths_folder = os.path.join(traj_folder, "depths")
                os.mkdir(frames_folder)
                os.mkdir(depths_folder)

                # Load the specific trajectory
                with open(trajectory, "r") as fopen:
                    waypoints = json.load(fopen)

                # Produce a photometric and depth rendering of each waypoint
                num_waypoints = len(list(waypoints.keys()))
                for wp in range(num_waypoints):
                    
                    # Sample a single waypoint c2w
                    c2w_wp = waypoints[str(wp)]
                    c2w_wp = torch.tensor(c2w_wp)

                    # Produce rays
                    test_rays = ray_gen(c2w_wp)

                    # Fix configuration code to 0 and produce images
                    ch_map, dp_map, _ = model(
                        rays=test_rays,
                        app_code=0
                    )

                    # Rebuild images and store in the epoch folder
                    frame = ch_map.detach().cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], cfg_parameters["channels"])
                    depth = dp_map.detach().cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], 1)
                    cv2.imwrite(os.path.join(frames_folder, f"{wp}.png"), frame)
                    cv2.imwrite(os.path.join(depths_folder, f"{wp}.png"), depth)


        # Call scheduler at the end of each epoch
        trainer_agent.scheduler.step()
        print(f"Current Learning Rate: {trainer_agent.scheduler.get_last_lr()[0]:.5f}")


# Main
if __name__ == "__main__":

    # Create a folder for this training session
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"training_logs/folder_{current_time}"
    os.makedirs(folder_name)

    # Launch training loop
    main(folder_name=folder_name)