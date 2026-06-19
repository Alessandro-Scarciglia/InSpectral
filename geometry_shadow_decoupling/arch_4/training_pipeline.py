# =======================================================================================
# IMPORT CUSTOM MODULES
# =======================================================================================
from data.rays_generator import RaysGeneratorSynth
from model.rendering import NeuralRenderer
from data.dataset import NeRFData
from trainer import Trainer
from model.metrics import *
from data.generate_postprocessed_dataset import calculate_intrinsic_matrix
from model.residuals_embedder import ResEmbedder

# =======================================================================================
# IMPORT GLOBAL HYPERPARAMETERS
# =======================================================================================
from config_parameters import *

# =======================================================================================
# IMPORT STANDARD PYTHON PACKAGES
# =======================================================================================
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  
import json
from datetime import datetime
import os
import time
from tqdm import tqdm
import re  

# Set random seeds to ensure deterministic behavior and experiment repeatability
torch.manual_seed(1234)


def main(folder_name: str) -> None:
    """
    Main training and validation pipeline for the joint NeRF and Pose Optimization (BARF-like).
    Handles data loading, ray deformation via residual embedding, rendering, and evaluation.
    """
    device = torch.device(cfg_parameters["device"])
    
    writer = SummaryWriter(log_dir=folder_name)
    print(f"[INFO] TensorBoard logging initialized at: {folder_name}")
    
    # -----------------------------------------------------------------------------------
    # DYNAMIC METADATA PATH SELECTION & ESTIMATION BOUNDS
    # -----------------------------------------------------------------------------------
    # Extract min and max numerical values from the dataset filename to align with pre-processing bounds
    filename = os.path.basename(dataset_parameters["data_path"])
    try:
        min_val = float(re.search(r"min(-?\d+\.\d+)", filename).group(1))
        max_val = float(re.search(r"max(-?\d+\.\d+)", filename).group(1))
        print(f"[GEOMETRY] Detected normalization bounds from dataset: Min={min_val}, Max={max_val}")
    except AttributeError:
        min_val, max_val = 23.40, 37.06
        print(f"[WARNING] Could not decode filename. Falling back to defaults: Min={min_val}, Max={max_val}")

    # Compute the isotropic scale factor as the inverse of the maximum absolute distance
    scale_factor = 1.0 / (max_val + 1e-10)
    
    # -----------------------------------------------------------------------------------
    # LOAD VALIDATION METADATA & CAMERA INTRINSICS
    # -----------------------------------------------------------------------------------
    test_json_path = os.path.join(dataset_parameters["test_path"], "transforms_test.json")
    if not os.path.exists(test_json_path):
        test_json_path = os.path.join(dataset_parameters["test_path"], "transforms_train.json")
        print(f"[WARNING] transforms_test.json not found. Falling back to: {test_json_path}")

    with open(test_json_path, "r") as test_fopen:
        test_df = json.load(test_fopen)
        test_samples = test_df["frames"]

    intrinsic_matrix = calculate_intrinsic_matrix(
        fov=test_df["camera_angle_x"],
        resolution=(cfg_parameters["resolution"], cfg_parameters["resolution"])
    )

    raygen = RaysGeneratorSynth(**rays_parameters, intrinsic_matrix=intrinsic_matrix)
    raygen = raygen.to(device)

    # -----------------------------------------------------------------------------------
    # INITIALIZE NEURAL ARCHITECTURES
    # -----------------------------------------------------------------------------------
    model = NeuralRenderer(
        **sampler_parameters, **hash_parameters, **sh_parameters, 
        **app_parameters, **nerf_parameters, device=cfg_parameters["device"]
    )

    res_model = ResEmbedder(**res_model_parameters, device=cfg_parameters["device"])

    if training_parameters["pretrained"]:
        print(f"Resuming pretrained NeRF model {training_parameters['checkpoint_nerf']}")
        checkpoint0 = torch.load(training_parameters["checkpoint_nerf"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint0["model_state_dict"])

        print(f"Resuming pretrained Residual model {training_parameters['checkpoint_res']}")
        checkpoint1 = torch.load(training_parameters["checkpoint_res"], map_location=device, weights_only=False)
        res_model.load_state_dict(checkpoint1["model_state_dict"])
        
    model.to(device)
    res_model.to(device)

    # -----------------------------------------------------------------------------------
    # PREPARE TRAINING DATALOADER
    # -----------------------------------------------------------------------------------
    training_dataset = NeRFData(dataset_parameters["data_path"])
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=training_parameters["training_batch"],
        shuffle=True,
        num_workers=4,
        pin_memory=True  
    )

    trainer_agent = Trainer(
        model=model, res_model=res_model,
        lr=training_parameters["lr"], lr_res=training_parameters["res_lr"],
        lr_app=training_parameters["app_lr"], lr_emb=training_parameters["emb_lr"],
        betas=training_parameters["betas"], eps=training_parameters["eps"],
        tv_loss_weight=training_parameters["tv_loss_weight"],
        sparsity_loss_weight=training_parameters["sparsity_loss_weight"],
        decay_rate=training_parameters["decay_rate"], decay_steps=training_parameters["decay_steps"],
        alpha=training_parameters["bce_loss_weight"], beta=training_parameters["dice_loss_weight"]
    )

    max_steps = training_parameters["epochs"] * len(training_dataloader)
    dl_len = len(training_dataloader)
    global_step = 0  
    
    # -----------------------------------------------------------------------------------
    # MAIN TRAINING LOOP
    # -----------------------------------------------------------------------------------
    for epoch in range(training_parameters["epochs"]):
        
        model.train()
        res_model.train()
        model.sampler.n_samples = sampler_parameters["n_ray_samples"]
        
        for n_iter, ray_batch in enumerate(training_dataloader):
            ray_batch = ray_batch.to(device, non_blocking=True)        

            if epoch >= training_parameters["start_cam_opt"]:
                d_rays_o, d_rays_rot = res_model(ray_batch[:, -1])
                orig = ray_batch[:, :3]      
                dirs = ray_batch[:, 3:6]     
                tail = ray_batch[:, 6:]      

                new_orig = orig + d_rays_o
                new_dirs = torch.einsum('bij,bj->bi', d_rays_rot, dirs) 
                ray_batch = torch.cat([new_orig, new_dirs, tail], dim=-1)

            losses = trainer_agent.train_one_batch(
                rays=ray_batch[:, :6], labels=ray_batch[:, 6:10], app_idx=ray_batch[:, 10:],
                epoch=epoch, n_iter=n_iter, max_steps=max_steps, dl_len=dl_len
            )
            
            loss, loss_photom, loss_tv, loss_sparsity, loss_segm, loss_app = losses
            
            writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Photometric_Loss", loss_photom.item(), global_step)
            writer.add_scalar("Train/TV_Loss", loss_tv.item(), global_step)
            writer.add_scalar("Train/Sparsity_Loss", loss_sparsity.item(), global_step)
            writer.add_scalar("Train/Segmentation_Loss", loss_segm.item(), global_step)
            writer.add_scalar("Train/Appearance_Loss", loss_app.item(), global_step)
            
            global_step += 1
                            
            if training_parameters["verbose"] and n_iter % 10 == 0:
                print(
                    f"[{epoch:02d}/{training_parameters['epochs']:02d}] "
                    f"Iter: {n_iter:04d}/{dl_len:04d} | "
                    f"Tot: {loss.item():.4f} | "
                    f"Photo: {loss_photom.item():.4f} | "
                    f"TV: {loss_tv.item():.4f} | "
                    f"Sparsity: {loss_sparsity.item():.4f} | "
                    f"Segm: {loss_segm.item():.4f} | "
                    f"App: {loss_app.item():.4f}"
                )

        # -----------------------------------------------------------------------------------
        # VALIDATION PHASE (EVALUATION & RENDERING)
        # -----------------------------------------------------------------------------------
        with torch.no_grad():
            
            model.eval()
            res_model.eval()

            test_full_psnr_set, test_obj_psnr_set, test_bkg_psnr_set = [], [], []
            test_full_mae_set, test_obj_mae_set, test_bkg_mae_set = [], [], []

            epoch_folder = os.path.join(folder_name, f"epoch_{epoch}")
            os.makedirs(epoch_folder, exist_ok=True)
            rgb_dst = os.path.join(epoch_folder, "rgb")
            depth_dst = os.path.join(epoch_folder, "depth")
            os.makedirs(rgb_dst, exist_ok=True)
            os.makedirs(depth_dst, exist_ok=True)

            for m_iter, test_sample in tqdm(enumerate(test_samples), desc=f"Validating Epoch {epoch}"):
                if m_iter % 10 != 0:
                    continue
                
                # Load uncompressed camera-to-world matrix from metadata
                test_c2w = torch.tensor(test_sample["transform_matrix"], dtype=torch.float32, device=device)
                
                # Apply pure linear scaling to the camera positions without centering offsets
                test_c2w[:3, 3] *= scale_factor

                # Generate camera rays inside the normalized coordinate space
                test_rays = raygen(test_c2w).reshape(-1, 6)
                
                num_rays = test_rays.shape[0]
                test_idx = torch.zeros((num_rays, 1), dtype=torch.float32, device=device)

                # Load and resize ground truth components safely
                img_path = os.path.join(dataset_parameters["test_path"], test_sample["file_path"])
                depth_path = os.path.join(dataset_parameters["test_path"], test_sample["depth_path"])
                mask_path = os.path.join(dataset_parameters["test_path"], test_sample["mask_path"])

                if not os.path.exists(img_path):
                    print(f"[ERROR] Image file not found at: {img_path}")
                    continue

                target_image = cv2.imread(img_path)
                target_image = cv2.resize(target_image, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
                target_image = torch.tensor(target_image, dtype=torch.float32, device=device).reshape(-1, cfg_parameters["channels"])

                target_depth = cv2.imread(depth_path, 0)
                target_depth = cv2.resize(target_depth, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
                target_depth = torch.tensor(target_depth, dtype=torch.float32, device=device).reshape(-1, 1)

                obj_mask = cv2.imread(mask_path, 0)
                obj_mask = cv2.resize(obj_mask, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
                obj_mask = torch.tensor(obj_mask, dtype=torch.float32, device=device).reshape(-1, 1)

                chunk_size = 4096 
                test_rgb_list, test_depth_list = [], []
                
                rays_chunks = torch.split(test_rays, chunk_size, dim=0)
                idx_chunks = torch.split(test_idx, chunk_size, dim=0)

                for r_chunk, i_chunk in zip(rays_chunks, idx_chunks):
                    rgb_n, depth_n, _, _ = model(
                        rays=r_chunk, 
                        app_idx=i_chunk, 
                        epoch=None, 
                        n_iter=None, 
                        max_steps=None, 
                        len_dl=None
                    )
                    test_rgb_list.append(rgb_n)
                    test_depth_list.append(depth_n)

                test_rgb = torch.cat(test_rgb_list, dim=0)
                test_depth = torch.cat(test_depth_list, dim=0)

                # Compute quantitative metrics
                test_full_psnr, test_obj_psnr, test_bkg_psnr = compute_psnr(
                    img1=test_rgb, img2=target_image, mask=obj_mask.repeat(1, cfg_parameters["channels"])
                )
                test_full_mae, test_obj_mae, test_bkg_mae = compute_mae(
                    img1=test_depth.unsqueeze(-1), img2=target_depth, mask=obj_mask
                )
            
                test_full_psnr_set.append(test_full_psnr if isinstance(test_full_psnr, float) else test_full_psnr.item())
                test_obj_psnr_set.append(test_obj_psnr if isinstance(test_obj_psnr, float) else test_obj_psnr.item())
                test_bkg_psnr_set.append(test_bkg_psnr if isinstance(test_bkg_psnr, float) else test_bkg_psnr.item())
                test_full_mae_set.append(test_full_mae if isinstance(test_full_mae, float) else test_full_mae.item())
                test_obj_mae_set.append(test_obj_mae if isinstance(test_obj_mae, float) else test_obj_mae.item())
                test_bkg_mae_set.append(test_bkg_mae if isinstance(test_bkg_mae, float) else test_bkg_mae.item())

                frame = test_rgb.cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], cfg_parameters["channels"])
                depth = test_depth.cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], 1)

                rgb_to_save = (frame * 255).astype(np.uint8)
                depth_to_save = (depth * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(rgb_dst, f"{m_iter:03}.png"), rgb_to_save)
                cv2.imwrite(os.path.join(depth_dst, f"{m_iter:03}.png"), depth_to_save)

            metrics_summary = {
                "psnr": np.mean(test_full_psnr_set), "mae": np.mean(test_full_mae_set),
                "psnr_obj": np.mean(test_obj_psnr_set), "mae_obj": np.mean(test_obj_mae_set),
                "psnr_bkg": np.mean(test_bkg_psnr_set), "mae_bkg": np.mean(test_bkg_mae_set)
            }

            writer.add_scalar("Val/Full_PSNR", metrics_summary["psnr"], epoch)
            writer.add_scalar("Val/Object_PSNR", metrics_summary["psnr_obj"], epoch)
            writer.add_scalar("Val/Background_PSNR", metrics_summary["psnr_bkg"], epoch)
            writer.add_scalar("Val/Full_MAE", metrics_summary["mae"], epoch)
            writer.add_scalar("Val/Object_MAE", metrics_summary["mae_obj"], epoch)
            writer.add_scalar("Val/Background_MAE", metrics_summary["mae_bkg"], epoch)

            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), **metrics_summary}
            checkpoint_res = {"epoch": epoch, "model_state_dict": res_model.state_dict(), **metrics_summary}
            
            torch.save(checkpoint, os.path.join(epoch_folder, "chkpt_nerf.pt"))
            torch.save(checkpoint_res, os.path.join(epoch_folder, "chkpt_res.pt"))
            
            if training_parameters["verbose"]:
                print(f"\nTest after epoch {epoch}:\n"
                      f"Avg. PSNR: {metrics_summary['psnr']:.3f} | Obj PSNR: {metrics_summary['psnr_obj']:.3f}\n"
                      f"Avg. MAE:  {metrics_summary['mae']:.3f} | Obj MAE:  {metrics_summary['mae_obj']:.3f}\n")

        trainer_agent.scheduler.step()

    writer.close()


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"geometry_shadow_decoupling/arch_4/results/folder_{current_time}"
    os.makedirs(folder_name, exist_ok=True)
    
    main(folder_name=folder_name)