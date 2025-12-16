# Import custom modules
from data.rays_generator import RaysGeneratorSynth
from model.rendering import NeuralRenderer
from data.dataset import NeRFData
from trainer import Trainer
from model.metrics import *
from data.generate_postprocessed_dataset import calculate_intrinsic_matrix

# Parameters
from config_parameters import *

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

# Set seeds for test repeatibility
torch.manual_seed(1234)


def main(
        folder_name: str
) -> None:
    """
    This function implements the training loop pipeline. It also includes the validation and test. This function is
    a wrap of a system which shall be distributed (i.e. it includes the acquisition of pictures, the preprocessing in
    camera rays, all the initializations, the optimization etc.)

    Parameters:
    ----------
    folder_name: str
        it is the output folder name for the full training. It is generate automatically with the
        datetime when the script is launched.
    """
    
    # Load the test dataset
    with open(dataset_parameters["test_path"] + "/transforms.json", "r") as train_fopen:
        test_df = json.load(train_fopen)
        test_samples = test_df["frames"]

    # Compute camera intrinsics for training
    intrinsic_matrix = calculate_intrinsic_matrix(
        fov=test_df["camera_angle_x"],
        resolution=(cfg_parameters["resolution"], cfg_parameters["resolution"])
    )

    # Define the rays generator object
    raygen = RaysGeneratorSynth(
        **rays_parameters,
        intrinsic_matrix=intrinsic_matrix
    )

    # Instantiate the model
    model = NeuralRenderer(
        **sampler_parameters,
        **hash_parameters,
        **sh_parameters,
        **app_parameters,
        **nerf_parameters,
        device=cfg_parameters["device"]
    )

    # Load Training and Validation datasets
    training_dataset = NeRFData(dataset_parameters["data_path"])

    # Create a dataloader for Training and Validation datasets
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
        decay_steps=training_parameters["decay_steps"],
        alpha=training_parameters["bce_loss_weight"],
        beta=training_parameters["dice_loss_weight"]
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

            # Train one batch 6+3+1+1
            losses = trainer_agent.train_one_batch(
                rays=ray_batch[:, :6],
                labels=ray_batch[:, 6:10],
                app_idx=ray_batch[:, 10:],
                epoch=epoch
            )

            # Split loss in loss components
            loss, loss_photom, loss_tv, loss_sparsity, loss_segm = losses
                            
            # Display loss values
            if training_parameters["verbose"]:
                print(f"Epoch: {epoch} | # Iter: {n_iter} | Elapsed time (s): {(time.time()-t_start):.3f} | "
                      f"Photometric Loss: {loss_photom.item():.5f} | TV Loss: {loss_tv.item():.5f} | Sparsity Loss: {loss_sparsity.item():.5f} | "
                      f"BCE-Dice Loss: {loss_segm.item():.5f} | "
                      f"Tot Loss: {loss.item():.5f}")

        # Validate epoch on the Validation dataset
        with torch.no_grad():

            # Create metrics buffer
            model.eval()
            test_full_psnr_set, test_obj_psnr_set, test_bkg_psnr_set = list(), list(), list()
            test_full_mae_set, test_obj_mae_set, test_bkg_mae_set = list(), list(), list()

            # Create the epoch folder
            epoch_folder = os.path.join(folder_name, f"epoch_{epoch}")
            os.mkdir(epoch_folder)

            # Create a folder for RGB maps and Depths
            rgb_dst = os.path.join(epoch_folder, "rgb")
            depth_dst = os.path.join(epoch_folder, "depth")
            os.mkdir(rgb_dst)
            os.mkdir(depth_dst)

            # Loop through the validation set
            for m_iter, test_sample in tqdm(enumerate(test_samples)):

                # Test 18 views out of 360
                if m_iter % 20:
                    continue
                
                # Generate rays and appearance idx
                test_c2w = torch.tensor(test_sample["transform_matrix"])
                #test_c2w[:3, 3] /= 10.
                test_rays = raygen(test_c2w).reshape(-1, 6)
                test_rays = test_rays.to(cfg_parameters["device"])
                test_idx = torch.zeros((cfg_parameters["resolution"] * cfg_parameters["resolution"], 1))

                # Retrieve labels
                target_image = cv2.imread(os.path.join(dataset_parameters["test_path"], test_sample["file_path"]))
                target_image = cv2.resize(target_image, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
                target_image = torch.tensor(target_image).reshape(-1, cfg_parameters["channels"]).to(cfg_parameters["device"])

                target_depth = cv2.imread(os.path.join(dataset_parameters["test_path"], test_sample["depth_path"]), 0)
                target_depth = cv2.resize(target_depth, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
                target_depth = torch.tensor(target_depth).reshape(-1, 1).to(cfg_parameters["device"])

                obj_mask = cv2.imread(os.path.join(dataset_parameters["test_path"], test_sample["mask_path"]), 0)
                obj_mask = cv2.resize(obj_mask, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
                obj_mask = torch.tensor(obj_mask).reshape(-1, 1).to(cfg_parameters["device"])

                # Estimate rendering: split the inference in N steps in chunks of 256x256.
                # e.g. Given 512, then 512/256=2 => n_iterations = (512/256)^2.
                # i.e. n_iter = (res//256)^2
                test_rgb, test_depth = list(), list()
                
                for n in range(((cfg_parameters["resolution"] // 256) ** 2)):
                    test_rgb_n, test_depth_n, _, _ = model(
                        test_rays[n * (256**2): (n+1) * (256**2), :],
                        test_idx[n * (256**2): (n+1) * (256**2), :]
                    )

                    test_rgb.append(test_rgb_n)
                    test_depth.append(test_depth_n)

                # Concatenate chunks
                test_rgb = torch.cat(test_rgb, dim=0).to(cfg_parameters["device"])
                test_depth = torch.cat(test_depth, dim=0).to(cfg_parameters["device"])

                # TODO: Mask the label image

                # Evaluate test PSNR and MAE
                test_full_psnr, test_obj_psnr, test_bkg_psnr = compute_psnr(img1=test_rgb, img2=target_image, mask=obj_mask.repeat(1, cfg_parameters["channels"]))
                test_full_mae, test_obj_mae, test_bkg_mae = compute_mae(img1=test_depth.unsqueeze(-1), img2=target_depth, mask=obj_mask)
            
                # Collect test batch metrics about PSNR and MAE
                test_full_psnr_set.append(test_full_psnr)
                test_obj_psnr_set.append(test_obj_psnr)
                test_bkg_psnr_set.append(test_bkg_psnr)

                test_full_mae_set.append(test_full_mae)
                test_obj_mae_set.append(test_obj_mae)
                test_bkg_mae_set.append(test_bkg_mae)

                # Reconstruct images and store them
                frame = test_rgb.detach().cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], cfg_parameters["channels"])
                depth = test_depth.detach().cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], 1)

                cv2.imwrite(os.path.join(rgb_dst, f"{m_iter:03}.png"), frame * 255)
                cv2.imwrite(os.path.join(depth_dst, f"{m_iter:03}.png"), depth * 255)

            # Store checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer_agent.optimizer.state_dict(),
                "psnr": np.mean(test_full_psnr_set),
                "mae": np.mean(test_full_mae_set),
                "psnr_obj": np.mean(test_obj_psnr_set),
                "mae_obj": np.mean(test_obj_mae_set),
                "psnr_bkg": np.mean(test_bkg_psnr_set),
                "mae_bkg": np.mean(test_bkg_mae_set)
            }
            
            torch.save(checkpoint, os.path.join(epoch_folder, "chkpt.pt"))
            
            # Print metrics
            if training_parameters["verbose"]:
                print(f"\nTest after epoch {epoch}:\n"
                      f"Avg. PSNR: {np.mean(test_full_psnr_set):.3f}  |  Avg. Object PSNR: {np.mean(test_obj_psnr_set):.3f}  |  Avg. Background PSNR: {np.mean(test_bkg_psnr_set):.3f}\n"
                      f"Avg.  MAE:  {np.mean(test_full_mae_set):.3f}  |  Avg. Object  MAE:  {np.mean(test_obj_mae_set):.3f}  |  Avg. Background  MAE:  {np.mean(test_bkg_mae_set):.3f}\n\n")

        # Call scheduler at the end of each epoch
        trainer_agent.scheduler.step()
        print(f"Current Learning Rate: {trainer_agent.scheduler.get_last_lr()[0]:.5f}")


if __name__ == "__main__":

    # Create a folder for this training session
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"geometry_shadow_decoupling/arch_4/results/folder_{current_time}"
    os.makedirs(folder_name)

    # Launch training loop
    main(folder_name=folder_name)
