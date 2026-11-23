# =======================================================================================
# IMPORT CUSTOM MODULES
# =======================================================================================
from data.rays_generator import RaysGeneratorSynth
from model.rendering import NeuralRenderer
from data.dataset import NeRFData
from training.trainer import Trainer
from model.metrics import *
from data.generate_postprocessed_dataset import calculate_intrinsic_matrix
from model.residuals_embedder import ResEmbedder

# =======================================================================================
# IMPORT GLOBAL HYPERPARAMETERS
# =======================================================================================
from configs.config_parameters import *

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
    # ENFORCE SYSTEM CHANNELS DIRECTLY FROM GLOBAL IMPORTED CONFIGURATION
    # -----------------------------------------------------------------------------------
    img_mode = dataset_parameters.get("mode", "rgb").lower()
    img_channels = cfg_parameters["channels"]  # Strictly derived from global config
    
    # Select path mapping keys and reading configurations based on the selected mode
    if img_mode == "rgb":
        path_key = "file_path"
        read_flag = cv2.IMREAD_COLOR if img_channels == 3 else cv2.IMREAD_GRAYSCALE
        print(f"[INFO] Configured Mode: RGB ({img_channels} channels, using '{path_key}')")
    elif img_mode == "ir":
        path_key = "ir_path"
        read_flag = cv2.IMREAD_GRAYSCALE if img_channels == 1 else cv2.IMREAD_COLOR
        print(f"[INFO] Configured Mode: IR ({img_channels} channels, using '{path_key}')")
    else:
        raise ValueError(f"Unknown mode: '{img_mode}'. Choose 'rgb' or 'ir'.")

    # -----------------------------------------------------------------------------------
    # DYNAMIC METADATA PATH SELECTION & ESTIMATION BOUNDS
    # -----------------------------------------------------------------------------------
    filename = os.path.basename(dataset_parameters["data_path"])
    try:
        min_val = float(re.search(r"min(-?\d+\.\d+)", filename).group(1))
        max_val = float(re.search(r"max(-?\d+\.\d+)", filename).group(1))
        print(f"[GEOMETRY] Detected normalization bounds from dataset: Min={min_val}, Max={max_val}")
    except AttributeError:
        min_val, max_val = 23.40, 37.06
        print(f"[WARNING] Could not decode filename. Falling back to defaults: Min={min_val}, Max={max_val}")

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
    renderer_kwargs = {}
    renderer_kwargs.update(sampler_parameters)
    renderer_kwargs.update(hash_parameters)
    renderer_kwargs.update(sh_parameters)
    renderer_kwargs.update(app_parameters)
    renderer_kwargs.update(nerf_parameters)
    renderer_kwargs["device"] = cfg_parameters["device"]

    model = NeuralRenderer(**renderer_kwargs)
    res_model = ResEmbedder(**res_model_parameters, device=cfg_parameters["device"])

    if training_parameters["pretrained"]:
        print(f"Resuming pretrained NeRF model {training_parameters['checkpoint_nerf']}")
        checkpoint0 = torch.load(training_parameters["checkpoint_nerf"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint0["model_state_dict"], strict=False)

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

    # Explicitly pass required parameters to satisfy the Trainer agent signature
    trainer_agent = Trainer(
        model=model, res_model=res_model,
        lr=training_parameters["lr"], lr_res=training_parameters["res_lr"],
        lr_app=training_parameters["app_lr"], lr_emb=training_parameters["emb_lr"],
        betas=training_parameters["betas"], eps=training_parameters["eps"],
        tv_loss_weight=training_parameters["tv_loss_weight"],
        photo_loss_weight=training_parameters["photometric_loss_weight"],
        sparsity_loss_weight=training_parameters["sparsity_loss_weight"],
        similarity_loss_weight=training_parameters["similarity_loss_weight"],
        decay_rate=training_parameters["decay_rate"], decay_steps=training_parameters["decay_steps"],
        softness=training_parameters["driving_loss_softness"],
        gamma=training_parameters["driving_loss_gamma"],
        stop_tv_epoch=training_parameters["stop_tv_epoch"],
        start_seg_epoch=training_parameters["start_seg_epoch"],
        driving_loss_weight=training_parameters["bce_dice_loss_weight"],
        enable_camera_refinement=training_parameters["enable_camera_refinement"],
        camera_warmup_epochs=training_parameters["camera_warmup_epochs"],
        camera_update_interval=training_parameters["camera_update_interval"],
        camera_only=training_parameters["camera_only"],
        camera_pose_reg_weight=training_parameters["camera_pose_reg_weight"],
        mixed_precision=training_parameters["mixed_precision"],
        distortion_loss_weight=training_parameters["distortion_loss_weight"]
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

            # Apply camera pose optimization via residual deformation if enabled for this epoch
            camera_enabled = (
                training_parameters["enable_camera_refinement"]
                and epoch >= training_parameters["camera_warmup_epochs"]
                and n_iter % max(1, training_parameters["camera_update_interval"]) == 0
            )
            if camera_enabled:
                d_rays_o, d_rays_rot = res_model(ray_batch[:, -1])
                orig = ray_batch[:, :3]      
                dirs = ray_batch[:, 3:6]     
                tail = ray_batch[:, 6:]      

                new_orig = orig + d_rays_o
                new_dirs = torch.einsum('bij,bj->bi', d_rays_rot, dirs) 
                ray_batch = torch.cat([new_orig, new_dirs, tail], dim=-1)

            # Perform dynamic vector slicing based on the chosen number of image channels
            # Rays (6) | Labels (img_channels) | Driving Mask (1) | Image Index (1)
            losses = trainer_agent.train_one_batch(
                rays=ray_batch[:, :6], 
                labels=ray_batch[:, 6 : 6 + img_channels],          
                driving_mask=ray_batch[:, 6 + img_channels : 6 + img_channels + 1],    
                app_idx=ray_batch[:, 6 + img_channels + 1 :],         
                epoch=epoch, n_iter=n_iter, max_steps=max_steps, dl_len=dl_len
            )
            
            loss, loss_photom, loss_tv, loss_sparsity, loss_driving, loss_app, loss_distortion = losses
            
            # Log metrics to Tensorboard
            writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Photometric_Loss", loss_photom.item(), global_step)
            writer.add_scalar("Train/TV_Loss", loss_tv.item(), global_step)
            writer.add_scalar("Train/Sparsity_Loss", loss_sparsity.item(), global_step)
            writer.add_scalar("Train/Driving_Loss", loss_driving.item(), global_step)
            writer.add_scalar("Train/Appearance_Loss", loss_app.item(), global_step)
            writer.add_scalar("Train/Distortion_Loss", loss_distortion.item(), global_step)
            
            global_step += 1
                            
            if training_parameters["verbose"] and n_iter % 10 == 0:
                print(
                    f"[{epoch:02d}/{training_parameters['epochs']:02d}] "
                    f"Iter: {n_iter:04d}/{dl_len:04d} | "
                    f"Tot: {loss.item():.4f} | "
                    f"Photo: {loss_photom.item():.4f} | "
                    f"TV: {loss_tv.item():.4f} | "
                    f"Sparsity: {loss_sparsity.item():.4f} | "
                    f"Drive: {loss_driving.item():.4f} | "
                    f"App: {loss_app.item():.4f}"
                )

        writer.flush()

        # -----------------------------------------------------------------------------------
        # VALIDATION PHASE (EVALUATION & RENDERING)
        # -----------------------------------------------------------------------------------
        with torch.no_grad():
            
            model.eval()
            res_model.eval()

            test_full_psnr_set, test_obj_psnr_set, test_bkg_psnr_set = [], [], []
            test_full_mae_set, test_obj_mae_set, test_bkg_mae_set = [], [], []
            test_full_ssim_set, test_obj_ssim_set, test_bkg_ssim_set = [], [], []

            # Set up output directories for saving rendered results
            epoch_folder = os.path.join(folder_name, f"epoch_{epoch}")
            os.makedirs(epoch_folder, exist_ok=True)
            rgb_dst = os.path.join(epoch_folder, "rgb")
            depth_dst = os.path.join(epoch_folder, "depth")
            os.makedirs(rgb_dst, exist_ok=True)
            os.makedirs(depth_dst, exist_ok=True)

            for m_iter, test_sample in tqdm(enumerate(test_samples), desc=f"Validating Epoch {epoch}"):
                if m_iter % 10 != 0:
                    continue
                
                # Apply scaling factor to the ground truth test trajectories
                test_c2w = torch.tensor(test_sample["transform_matrix"], dtype=torch.float32, device=device)
                test_c2w[:3, 3] *= scale_factor

                test_rays = raygen(test_c2w).reshape(-1, 6)
                
                num_rays = test_rays.shape[0]
                test_idx = torch.ones((num_rays, 1), dtype=torch.float32, device=device) * (m_iter // 4)

                # Build dynamic evaluation paths using the selected dictionary keys
                img_path = os.path.join(dataset_parameters["test_path"], test_sample[path_key])
                depth_path = os.path.join(dataset_parameters["test_path"], test_sample["depth_path"])
                mask_path = os.path.join(dataset_parameters["test_path"], test_sample["mask_path"])

                if not os.path.exists(img_path):
                    print(f"[ERROR] Image file not found at: {img_path}")
                    continue

                # Load target images using the appropriate OpenCV read flag
                target_image = cv2.imread(img_path, read_flag)
                target_image = cv2.resize(target_image, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
                
                # Restore the channel dimension if cv2.resize removed it for single-channel configurations
                if len(target_image.shape) == 2 or target_image.shape[-1] != img_channels:
                    target_image = target_image.reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], -1)[:, :, :img_channels]
                    
                target_image = torch.tensor(target_image, dtype=torch.float32, device=device).reshape(-1, img_channels)

                # Load depth maps and evaluation masks
                target_depth = cv2.imread(depth_path, 0)
                target_depth = cv2.resize(target_depth, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
                target_depth = torch.tensor(target_depth, dtype=torch.float32, device=device).reshape(-1, 1)

                obj_mask = cv2.imread(mask_path, 0)
                obj_mask = cv2.resize(obj_mask, (cfg_parameters["resolution"], cfg_parameters["resolution"])) / 255. 
                obj_mask = torch.tensor(obj_mask, dtype=torch.float32, device=device).reshape(-1, 1)

                # Query the model using explicit data chunks to optimize GPU memory footprint
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

                # Explicitly slice model output to match the configuration channel settings
                if test_rgb.shape[-1] != img_channels:
                    test_rgb = test_rgb[:, :img_channels]

                # Compute quantitative evaluation metrics matching the configured channel depth
                test_full_psnr, test_obj_psnr, test_bkg_psnr = compute_psnr(
                    img1=test_rgb, img2=target_image, mask=obj_mask.repeat(1, img_channels)
                )
                test_full_mae, test_obj_mae, test_bkg_mae = compute_mae(
                    img1=test_depth.unsqueeze(-1), img2=target_depth, mask=obj_mask
                )
                
                test_full_ssim, test_obj_ssim, test_bkg_ssim = compute_ssim(
                    img1=test_rgb, img2=target_image, mask=obj_mask,
                    resolution=cfg_parameters["resolution"], channels=img_channels
                )
            
                test_full_psnr_set.append(test_full_psnr if isinstance(test_full_psnr, float) else test_full_psnr.item())
                test_obj_psnr_set.append(test_obj_psnr if isinstance(test_obj_psnr, float) else test_obj_psnr.item())
                test_bkg_psnr_set.append(test_bkg_psnr if isinstance(test_bkg_psnr, float) else test_bkg_psnr.item())
                test_full_mae_set.append(test_full_mae if isinstance(test_full_mae, float) else test_full_mae.item())
                test_obj_mae_set.append(test_obj_mae if isinstance(test_obj_mae, float) else test_obj_mae.item())
                test_bkg_mae_set.append(test_bkg_mae if isinstance(test_bkg_mae, float) else test_bkg_mae.item())
                
                test_full_ssim_set.append(test_full_ssim.item())
                test_obj_ssim_set.append(test_obj_ssim.item())
                test_bkg_ssim_set.append(test_bkg_ssim.item())

                # Prepare predictions for file system output
                frame = test_rgb.cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], img_channels)
                depth = test_depth.cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], 1)

                rgb_to_save = (frame * 255).astype(np.uint8)
                depth_to_save = (depth * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(rgb_dst, f"{m_iter:03}.png"), rgb_to_save)
                cv2.imwrite(os.path.join(depth_dst, f"{m_iter:03}.png"), depth_to_save)
                
                # Adapt TensorBoard visualization layout depending on image color dimensionality
                if img_channels == 3:
                    tb_rgb = cv2.cvtColor(rgb_to_save, cv2.COLOR_BGR2RGB)
                    tb_rgb = torch.tensor(tb_rgb, dtype=torch.uint8).permute(2, 0, 1)
                else:
                    tb_rgb = torch.tensor(rgb_to_save, dtype=torch.uint8).permute(2, 0, 1)
                    
                tb_depth = torch.tensor(depth_to_save, dtype=torch.uint8).permute(2, 0, 1)
                
                writer.add_image(f"Val_Images/RGB_Sample_{m_iter:03d}", tb_rgb, epoch)
                writer.add_image(f"Val_Images/Depth_Sample_{m_iter:03d}", tb_depth, epoch)

            # Gather comprehensive performance summaries across all test viewpoints
            metrics_summary = {
                "psnr": np.mean(test_full_psnr_set), "mae": np.mean(test_full_mae_set), "ssim": np.mean(test_full_ssim_set),
                "psnr_obj": np.mean(test_obj_psnr_set), "mae_obj": np.mean(test_obj_mae_set), "ssim_obj": np.mean(test_obj_ssim_set),
                "psnr_bkg": np.mean(test_bkg_psnr_set), "mae_bkg": np.mean(test_bkg_mae_set), "ssim_bkg": np.mean(test_bkg_ssim_set)
            }

            writer.add_scalar("Val/Full_PSNR", metrics_summary["psnr"], epoch)
            writer.add_scalar("Val/Object_PSNR", metrics_summary["psnr_obj"], epoch)
            writer.add_scalar("Val/Background_PSNR", metrics_summary["psnr_bkg"], epoch)
            writer.add_scalar("Val/Full_MAE", metrics_summary["mae"], epoch)
            writer.add_scalar("Val/Object_MAE", metrics_summary["mae_obj"], epoch)
            writer.add_scalar("Val/Background_MAE", metrics_summary["mae_bkg"], epoch)
            writer.add_scalar("Val/Full_SSIM", metrics_summary["ssim"], epoch)
            writer.add_scalar("Val/Object_SSIM", metrics_summary["ssim_obj"], epoch)
            writer.add_scalar("Val/Background_SSIM", metrics_summary["ssim_bkg"], epoch)

            writer.flush()

            # Save network model states and optimization weights
            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), **metrics_summary}
            checkpoint_res = {"epoch": epoch, "model_state_dict": res_model.state_dict(), **metrics_summary}
            
            torch.save(checkpoint, os.path.join(epoch_folder, "chkpt_nerf.pt"))
            torch.save(checkpoint_res, os.path.join(epoch_folder, "chkpt_res.pt"))
            
            if training_parameters["verbose"]:
                print(f"\nTest after epoch {epoch}:\n"
                      f"Avg. PSNR: {metrics_summary['psnr']:.3f} | Obj PSNR: {metrics_summary['psnr_obj']:.3f}\n"
                      f"Avg. SSIM: {metrics_summary['ssim']:.3f} | Obj SSIM: {metrics_summary['ssim_obj']:.3f}\n"
                      f"Avg. MAE:  {metrics_summary['mae']:.3f} | Obj MAE:  {metrics_summary['mae_obj']:.3f}\n")

        # Step the learning-rate scheduler once per epoch
        trainer_agent.scheduler.step()


# =======================================================================================
# EXECUTION ENTRY POINT
# =======================================================================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    output_directory = os.path.join(
        "results", 
        f"folder_{timestamp}"
    )
    
    main(output_directory)