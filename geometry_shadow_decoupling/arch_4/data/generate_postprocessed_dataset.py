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
import matplotlib.pyplot as plt  

# =======================================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =======================================================================================
CONFIG = {
    "target_resolution": 512,
    "src_dataset": "Landsat8_v4",
    "arch": "arch_4",
    "deg_res": 4,
    
    # NUOVO PARAMETRO: Scegli 'rgb' (carica file_path, 3 canali) oppure 'ir' (carica ir_path, 1 canale)
    "image_type": "rgb", 
    
    # AGGIORNATO: Specifica il percorso relativo interno alla cartella del dataset
    "masked": "images/mask_sam2_grounded_bi",  
    "trajectory_type": "nominal",
    
    # Input path setup
    "data_path": "/home/vision/Desktop/Datasets/Landsat8_v4/",
    
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
    c2w_matrices = np.array([sample["transform_matrix"] for sample in samples])
    original_centers = c2w_matrices[:, :3, 3].copy()  
    
    distances = np.linalg.norm(original_centers, axis=1)
    min_dist = float(distances.min())
    max_dist = float(distances.max())
    
    scale_factor = 1.0 / (max_dist + 1e-10)
    print(f"[INFO] Trajectory Bounds from Origin - Min Dist: {min_dist:.4f}, Max Dist: {max_dist:.4f}")

    normalized_centers = original_centers * scale_factor

    # -----------------------------------------------------------------------------------
    # PARAMETRIC CONFIGURATION (RGB vs IR) BASED ON USER CONFIG
    # -----------------------------------------------------------------------------------
    image_mode = CONFIG["image_type"].lower()
    if image_mode == "rgb":
        path_key = "file_path"
        img_channels = 3
        read_flag = cv2.IMREAD_COLOR
        print("[INFO] Configured mode: RGB (3 channels, using 'file_path')")
    elif image_mode == "ir":
        path_key = "ir_path"
        img_channels = 1
        read_flag = cv2.IMREAD_GRAYSCALE
        print("[INFO] Configured mode: IR / Grayscale (1 channel, using 'ir_path')")
    else:
        raise ValueError(f"Unknown image_type: '{CONFIG['image_type']}'. Choose 'rgb' or 'ir'.")

    # -----------------------------------------------------------------------------------
    # MODEL & FILENAME SETUP
    # -----------------------------------------------------------------------------------
    intrinsic_matrix = calculate_intrinsic_matrix(fov=fov, resolution=(resolution, resolution))
    raygen = RaysGeneratorSynth(height=resolution, width=resolution, num_ch=num_ch, intrinsic_matrix=intrinsic_matrix)

    # FIX: Sostituiamo gli slash con underscore per non creare percorsi fittizi nel file name
    safe_masked_name = CONFIG['masked'].replace("/", "_")

    # Dynamic filename containing trajectory type info, min and max distance details from CONFIG
    base_name = f"{CONFIG['src_dataset']}_{CONFIG['deg_res']}_{CONFIG['trajectory_type']}_{safe_masked_name}_{CONFIG['arch']}_res{resolution}_min{min_dist:.2f}_max{max_dist:.2f}_{image_mode}"
    filename = f"{base_name}.npy"
    plot_filename = f"{base_name}_trajectory.png"
    
    os.makedirs(CONFIG["data_dst_dir"], exist_ok=True)
    data_dst = os.path.join(CONFIG["data_dst_dir"], filename)
    plot_dst = os.path.join(CONFIG["data_dst_dir"], plot_filename)

    # Allocazione dinamica delle colonne: 6 (rays) + n_canali (1 o 3) + 1 (Mask) + 1 (Image Index)
    total_features = 6 + img_channels + 1 + 1
    dataset = np.zeros((total_pixels, total_features), dtype=np.float32)

    # -----------------------------------------------------------------------------------
    # VECTORIZED DATASET GENERATION
    # -----------------------------------------------------------------------------------
    print("Generating Training Dataset...")
    for idx, sample in enumerate(tqdm(samples)):

        # Caricamento guidato dal parametro configurato
        if path_key not in sample:
            raise KeyError(f"[ERROR] Chiave '{path_key}' non trovata nel file JSON per il frame {idx}. Verifica i dati o la CONFIG.")
            
        img_path = os.path.join(CONFIG["data_path"], sample[path_key])
        img = cv2.imread(img_path, read_flag)
        
        if img is None:
            raise FileNotFoundError(f"[ERROR] Non è stato possibile caricare l'immagine al percorso: {img_path}")
            
        img = cv2.resize(img, (resolution, resolution))
        
        # Ripristina la dimensione del canale se OpenCV la rimuove per la scala di grigi (1 canale)
        if img_channels == 1:
            img = img[:, :, np.newaxis]
        
        # FIX: Isola il nome del file della maschera per evitare percorsi duplicati (es. mask/mask/000.png)
        mask_filename = os.path.basename(sample["mask_path"])
        dynamic_mask_path = os.path.join(CONFIG["masked"], mask_filename)
            
        mask_path = os.path.join(CONFIG["data_path"], dynamic_mask_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Controllo di sicurezza preventivo se il file non viene trovato
        if mask is None:
            raise FileNotFoundError(f"[ERROR] Non è stato possibile caricare la maschera al percorso: {mask_path}")
            
        mask = cv2.resize(mask, (resolution, resolution))

        # Forza a nero i pixel dello sfondo esterno alla maschera
        img = cv2.bitwise_and(img, img, mask=mask)
        
        # Di sicurezza dopo l'operazione bitwise su immagini a singolo canale
        if img_channels == 1 and len(img.shape) == 2:
            img = img[:, :, np.newaxis]

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

        # Concatenazione dinamica in base alle colonne configurate
        frame_data = np.concatenate([rays, img_norm, mask_norm, image_index], axis=-1)
        frame_data = frame_data.reshape(-1, total_features)

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
    print(f"Dataset successfully saved to: {data_dst} | Final Shape: {dataset.shape}")


if __name__ == "__main__":
    # Ensure local module visibility before runtime execution
    from rays_generator import RaysGeneratorSynth
    main()