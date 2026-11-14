"""
Elaboration of training output.
This script loads validation checkpoints across epochs for two distinct training
runs (with and without segmentation/shadow decoupling), extracts MAE metrics, 
and generates a comparative evaluation plot.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# =======================================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =======================================================================================
CONFIG = {
    "num_epochs": 15,
    "scale_factor": 100.0,  # Multiplier to convert MAE to percentage or relative scale
    
    # Base paths for the two experiments
    "path_without_seg": "geometry_shadow_decoupling/arch_4/results/SPAICE2026/test_32_4",
    "path_with_seg": "geometry_shadow_decoupling/arch_4/results/SPAICE2026/test_32_4_seg",
    
    # Filename structure
    "checkpoint_template": "epoch_{epoch}/chkpt.pt",
    
    # Plotting configuration
    "plot_title": "Relative MAE on Depth",
    "y_label": "Relative MAE [%]",
    "fig_size": (8, 5)
}


def load_experiment_metrics(base_path: str, num_epochs: int, scale: float) -> tuple[list[float], list[float], list[float]]:
    """
    Helper function to iterate through epochs and extract metrics from checkpoints.
    """
    mae_obj, mae_bkg, mae_full = [], [], []
    
    for epoch in range(num_epochs):
        # Dynamically compose the file path using the configuration templates
        chkpt_rel_path = CONFIG["checkpoint_template"].format(epoch=epoch)
        src_path = os.path.join(base_path, chkpt_rel_path)
        
        if not os.path.exists(src_path):
            print(f"[WARNING] Checkpoint not found at: {src_path}. Skipping remaining epochs.")
            break
            
        checkpoint = torch.load(src_path, map_location="cpu", weights_only=False)
        
        # Extract and scale metrics safely
        mae_obj.append(checkpoint.get("mae_obj", 0.0) * scale)
        mae_bkg.append(checkpoint.get("mae_bkg", 0.0) * scale)
        mae_full.append(checkpoint.get("mae", 0.0) * scale)
        
    return mae_obj, mae_bkg, mae_full


def main():
    # -----------------------------------------------------------------------------------
    # METRIC EXTRACTION
    # -----------------------------------------------------------------------------------
    print("[INFO] Loading metrics for experiment WITHOUT segmentation...")
    mae_obj, mae_bkg, _ = load_experiment_metrics(
        base_path=CONFIG["path_without_seg"], 
        num_epochs=CONFIG["num_epochs"], 
        scale=CONFIG["scale_factor"]
    )
    
    print("[INFO] Loading metrics for experiment WITH segmentation...")
    mae_obj_seg, mae_bkg_seg, _ = load_experiment_metrics(
        base_path=CONFIG["path_with_seg"], 
        num_epochs=CONFIG["num_epochs"], 
        scale=CONFIG["scale_factor"]
    )

    # -----------------------------------------------------------------------------------
    # PLOT GENERATION & VISUALIZATION
    # -----------------------------------------------------------------------------------
    plt.figure(figsize=CONFIG["fig_size"])
    plt.title(CONFIG["plot_title"], fontsize=12, fontweight='bold')
    
    # Plot baseline experiment (solid lines)
    plt.plot(mae_bkg, label="Background w/o", linestyle='-', color='tab:blue')
    plt.plot(mae_obj, label="SRO w/o", linestyle='-', color='tab:orange')
    
    # Plot segmented experiment (dashed lines with matching colors)
    plt.plot(mae_bkg_seg, label="Background w", linestyle='--', color='tab:blue', alpha=0.8)
    plt.plot(mae_obj_seg, label="SRO w", linestyle='--', color='tab:orange', alpha=0.8)
    
    # Layout and labels styling
    plt.ylabel(CONFIG["y_label"], fontsize=10)
    plt.xlabel("Epochs", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Reposition legend below the chart area to avoid overlapping data curves
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=True
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()