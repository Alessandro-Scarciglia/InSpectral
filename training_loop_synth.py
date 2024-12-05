'''
This file is meant to build the training process of the rendering model.
'''

# Import custom modules
from rays_generator_synth import RaysGeneratorSynth
from rendering import NeuralRenderer
from dataset_synth import NeRFData
from trainer import Trainer
from metrics import *

# Parameters
from parameters_synth import *

# Import standard modules
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import json
from datetime import datetime
import os
import time
from tqdm import tqdm

# Parameters
TEST_DATA = "data/nerf_synthetic/lego"

# Set seeds for test repeatibility
torch.manual_seed(1234)


# Training function
def main(folder_name: str):
    
    # Load data for testing
    with open(TEST_DATA + "/transforms_test.json", "r") as train_fopen:
        test_df = json.load(train_fopen)
        test_samples = test_df["frames"]

    # Load calibration matrix
    with open("calibration/calibration_synth.json", "r") as fopen:
        K = json.load(fopen)["mtx"]
        K = np.array(K).reshape(3, 3)
        K[:2, :3] /= 2.

    # Define the rays generator object
    raygen = RaysGeneratorSynth(**rays_parameters, K=K)

    # Instantiate the model
    model = NeuralRenderer(
        **sampler_parameters,
        **hash_parameters,
        **posenc_parameters,
        **sh_parameters,
        **nerf_parameters,
        device=cfg_parameters["device"]
    )

    # Load training and validation datasets
    training_dataset = NeRFData(**dataset_parameters)

    # Create a dataloader for training and validation datasets
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=training_parameters["training_batch"],
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
            
            # Bring batch to target device
            ray_batch = ray_batch.to(cfg_parameters["device"])

            # Train one batch
            losses = trainer_agent.train_one_batch(
                rays=ray_batch[:, :6],
                labels=ray_batch[:, 6:],
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
            test_psnr_set = []

            # Create the epoch folder
            epoch_folder = os.path.join(folder_name, f"epoch_{epoch}")
            os.mkdir(epoch_folder)

            # Create a folder for RGB maps and Depths
            rgb_dst = os.path.join(epoch_folder, "rgb")
            depth_dst = os.path.join(epoch_folder, "depth")
            os.mkdir(rgb_dst)
            os.mkdir(depth_dst)
            
            # Loop through the valtestidation set
            for m_iter, test_sample in tqdm(enumerate(test_samples)):
                
                # Test 25 views out of 200
                if m_iter % 8:
                    continue
                
                # Generate rays
                test_c2w = torch.tensor(test_sample["transform_matrix"])
                test_rays = raygen(test_c2w).reshape(-1, 6)
                test_rays = test_rays.to(cfg_parameters["device"])

                # Retrieve labels
                target_image = cv2.imread(os.path.join(TEST_DATA, test_sample["file_path"] + ".png"))
                target_image = cv2.resize(target_image, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255.
                target_image = torch.tensor(target_image).reshape(-1, 3).to(cfg_parameters["device"])      

                # Estimate rendering in 4 batches 
                # test_rgb = torch.tensor([], device=cfg_parameters["device"])
                # test_depth = torch.tensor([], device=cfg_parameters["device"])
                # for n_patch in range(4):
                #     rgb_patch, depth_patch, _ = model(test_rays[n_patch*(200**2): (n_patch+1)*(200**2)])
                #     test_rgb = torch.concatenate([test_rgb, rgb_patch], dim=0)
                #     test_depth = torch.concatenate([test_depth, depth_patch], dim=0)
                test_rgb, test_depth, _ = model(test_rays)
                
                # Evaluate test PSNR
                test_psnr = compute_psnr(img1=test_rgb, img2=target_image)

                # Split validation output
                test_psnr_set.append(test_psnr)

                # Reconstruct images and store them
                frame = test_rgb.detach().cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], cfg_parameters["channels"])
                depth = test_depth.detach().cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], 1)

                cv2.imwrite(os.path.join(rgb_dst, f"{m_iter:02}.png"), frame * 255)
                cv2.imwrite(os.path.join(depth_dst, f"{m_iter:02}.png"), depth * 255)


            # Store checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer_agent.optimizer.state_dict(),
                "psnr": np.mean(test_psnr_set)
            }
            
            torch.save(checkpoint, os.path.join(epoch_folder, "chkpt.pt"))
            
            # Print metrics
            if training_parameters["verbose"]:
                print(f"Test after epoch {epoch} | avg_PSNR: {np.mean(test_psnr_set):.3f}\n\n")


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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      