"""
This script transforms the images and labels of a custom NeRF Dataset to produce another dataset directly 
injectable in NeRF training. Everything is transformed in a numpy output file .npy where each sample refers to a single pixel.
Includes a final 3D visualization of original vs normalized camera trajectories.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt  # Added for plotting

# =======================================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =======================================================================================
CONFIG = {
    "target_resolution": 512,
    "src_dataset": "Landsat",
    "arch": "arch_4",
    "deg_res": 4,
    "masked": "masked",
    "trajectory_type": "nominal",  # Options: "estimated" or "nominal"
    
    # Input path setup
    "data_path": "/home/vision/Desktop/Datasets/Landsat/",
    
    # Output path setup
    "data_dst_dir": "/home/vision/Desktop/InSpectral/geometry_shadow_decoupling/arch_4/data/postprocessed_dataset/"
}


def calculate_intrinsic_matrix(fov: float, resolution: tuple[int, int]) -> np.ndarray:
    """Calculate the intrinsic camera matrix K for an ideal pinhole camera."""
    width, height = resolution
    fx = width / (2 * np.tan(fov / 2))
    fy = height / (2 * np.tan(fov / 2))
    cx = width / 2.
    cy = height / 2.
    
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])


def plot_trajectory(original_centers: np.ndarray, normalized_centers: np.ndarray, save_path: str):
    """Generates a 3D plot comparing the original trajectory with the unit-box normalized one,
    ensuring all axes share the exact same scale to prevent geometric distortion."""
    fig = plt.figure(figsize=(14, 6))
    
    # -----------------------------------------------------------------------------------
    # SUBPLOT 1: ORIGINAL TRAJECTORY (WITH EQUAL SCALING)
    # -----------------------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(original_centers[:, 0], original_centers[:, 1], original_centers[:, 2], 'b-', label='Trajectory Path')
    ax1.scatter(original_centers[:, 0], original_centers[:, 1], original_centers[:, 2], c=np.arange(len(original_centers)), cmap='viridis', s=15)
    
    # Calculate the max range across all dimensions to force equal scaling
    X, Y, Z = original_centers[:, 0], original_centers[:, 1], original_centers[:, 2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax1.set_title("Original Camera Trajectory (True Scale)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.grid(True)
    
    # -----------------------------------------------------------------------------------
    # SUBPLOT 2: NORMALIZED TRAJECTORY (WITH EQUAL SCALING [-1.1, 1.1])
    # -----------------------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(normalized_centers[:, 0], normalized_centers[:, 1], normalized_centers[:, 2], 'r-', label='Normalized Path')
    ax2.scatter(normalized_centers[:, 0], normalized_centers[:, 1], normalized_centers[:, 2], c=np.arange(len(normalized_centers)), cmap='viridis', s=15)
    
    # Force strict symmetric limits for the normalized cubic box
    box_lim = 1.1
    ax2.set_xlim([-box_lim, box_lim])
    ax2.set_ylim([-box_lim, box_lim])
    ax2.set_zlim([-box_lim, box_lim])
    
    ax2.set_title("Normalized Trajectory (Pure Linear Scaling)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.grid(True)
    
    # Matplotlib 3.x configuration to reinforce uniform bounding box proportions
    ax1.set_box_aspect((1, 1, 1))
    ax2.set_box_aspect((1, 1, 1))
    
    plt.suptitle(f"Camera Trajectory Analysis - Type: {CONFIG['trajectory_type'].upper()}", fontsize=14)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=200)
    print(f"[INFO] Trajectory plot successfully saved to: {save_path}")
    plt.close()


def main():
    """Pipeline to produce and store the postprocessed datasets, from NeRF images/metadata to rays."""
    
    # -----------------------------------------------------------------------------------
    # DYNAMIC METADATA PATH SELECTION
    # -----------------------------------------------------------------------------------
    if CONFIG["trajectory_type"] == "estimated":
        json_filename = "transforms_est.json"
    elif CONFIG["trajectory_type"] == "nominal":
        json_filename = "transforms_train.json"
    else:
        raise ValueError(f"Unknown trajectory_type: '{CONFIG['trajectory_type']}'. Choose 'estimated' or 'nominal'.")

    json_path = os.path.join(CONFIG["data_path"], json_filename)
    print(f"[INFO] Loading metadata from: {json_path}")
    
    with open(json_path, "r") as fopen:
        df = json.load(fopen)
        fov = df["camera_angle_x"]
        num_ch = int(df["num_ch"])
        all_samples = df["frames"]

    # Use the custom target resolution directly from CONFIG
    resolution = CONFIG["target_resolution"]
    num_frames = len(all_samples[::CONFIG["deg_res"]])
    pixels_per_frame = resolution * resolution
    total_pixels = num_frames * pixels_per_frame

    # Filter samples based on the configured stride upfront
    samples = all_samples[::CONFIG["deg_res"]]

    print(f"[INFO] Processing {num_frames} frames out of {len(all_samples)} total frames at resolution {resolution}x{resolution}.")

    # -----------------------------------------------------------------------------------
    # PURE LINEAR SCALING CALCULATION (NO OFFSET)
    # -----------------------------------------------------------------------------------
    # Extract all camera centers to compute scaling bounds directly from the original origin
    c2w_matrices = np.array([sample["transform_matrix"] for sample in samples])
    original_centers = c2w_matrices[:, :3, 3].copy()  
    
    # Calculate distances directly from the original coordinate origin (0, 0, 0)
    distances = np.linalg.norm(original_centers, axis=1)
    min_dist = float(distances.min())
    max_dist = float(distances.max())
    
    # Isotropic scale factor based on maximum absolute distance from the absolute origin
    scale_factor = 1.0 / (max_dist + 1e-10)
    print(f"[INFO] Trajectory Bounds from Origin - Min Dist: {min_dist:.4f}, Max Dist: {max_dist:.4f}")

    # Scale original centers directly without centering offsets
    normalized_centers = original_centers * scale_factor

    # -----------------------------------------------------------------------------------
    # MODEL & FILENAME SETUP
    # -----------------------------------------------------------------------------------
    intrinsic_matrix = calculate_intrinsic_matrix(fov=fov, resolution=(resolution, resolution))
    raygen = RaysGeneratorSynth(height=resolution, width=resolution, num_ch=num_ch, intrinsic_matrix=intrinsic_matrix)

    # Dynamic filename containing trajectory type info, min and max distance details from CONFIG
    base_name = f"{CONFIG['src_dataset']}_{CONFIG['deg_res']}_{CONFIG['trajectory_type']}_{CONFIG['masked']}_{CONFIG['arch']}_res{resolution}_min{min_dist:.2f}_max{max_dist:.2f}"
    filename = f"{base_name}.npy"
    plot_filename = f"{base_name}_trajectory.png"
    
    os.makedirs(CONFIG["data_dst_dir"], exist_ok=True)
    data_dst = os.path.join(CONFIG["data_dst_dir"], filename)
    plot_dst = os.path.join(CONFIG["data_dst_dir"], plot_filename)

    # Pre-allocate large NumPy array for structural efficiency (11 features per pixel)
    dataset = np.zeros((total_pixels, 11), dtype=np.float32)

    # -----------------------------------------------------------------------------------
    # VECTORIZED DATASET GENERATION
    # -----------------------------------------------------------------------------------
    print("Generating Training Dataset...")
    for idx, sample in enumerate(tqdm(samples)):

        # Load and resize image & mask using configured data path and the custom resolution
        img_path = os.path.join(CONFIG["data_path"], sample["file_path"])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (resolution, resolution))
        
        mask_path = os.path.join(CONFIG["data_path"], sample["mask_path"])
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (resolution, resolution))

        if CONFIG["masked"] == 'masked':
            img = cv2.bitwise_and(img, img, mask=mask)

        # Normalize components to [0, 1] range
        img_norm = img.astype(np.float32) / 255.0
        mask_norm = mask.astype(np.float32)[:, :, np.newaxis] / 255.0

        # Extract camera matrix and apply pure linear scaling to translation vectors
        c2w = np.array(sample["transform_matrix"], dtype=np.float32)
        c2w[:3, 3] *= scale_factor
        
        # Generate rays using the normalized pose matrix
        rays = raygen(torch.tensor(c2w)).numpy()

        # Create the index tracking grid
        image_index = np.full((resolution, resolution, 1), idx, dtype=np.float32)

        # Linearly combine all grids per pixel
        frame_data = np.concatenate([rays, img_norm, mask_norm, image_index], axis=-1)
        frame_data = frame_data.reshape(-1, 11)

        # Store directly inside the pre-allocated slice
        start_idx = idx * pixels_per_frame
        end_idx = start_idx + pixels_per_frame
        dataset[start_idx:end_idx] = frame_data

    # -----------------------------------------------------------------------------------
    # VISUALIZATION BLOCK (PLOT)
    # -----------------------------------------------------------------------------------
    plot_trajectory(original_centers=original_centers, normalized_centers=normalized_centers, save_path=plot_dst)

    # -----------------------------------------------------------------------------------
    # SHUFFLE & SAVE
    # -----------------------------------------------------------------------------------
    print(f"Shuffling and saving {num_frames} processed frames...")
    np.random.shuffle(dataset)
    np.save(data_dst, dataset)
    print(f"Dataset successfully saved to: {data_dst}")


if __name__ == "__main__":
    # Ensure local module visibility before runtime execution
    from rays_generator import RaysGeneratorSynth
    main()