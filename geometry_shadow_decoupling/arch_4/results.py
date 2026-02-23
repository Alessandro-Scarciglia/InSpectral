"""
Elaboration of training output.
"""

# Import modules
import torch
import matplotlib.pyplot as plt
import numpy as np

# Define scores
mae_obj, mae_bkg, mae_full = list(), list(), list()
mae_obj_seg, mae_bkg_seg, mae_full_seg = list(), list(), list()

# Iterate through epochs
for i in range(15):
    SRC_PATH = f"geometry_shadow_decoupling/arch_4/results/test_32_4/epoch_{i}/chkpt.pt"
    checkpoint = torch.load(
        SRC_PATH,
        map_location="cpu",
        weights_only=False
    )

    # Gather epoch metrics
    mae_obj.append(checkpoint["mae_obj"]*100)
    mae_bkg.append(checkpoint["mae_bkg"]*100)
    mae_full.append(checkpoint["mae"]*100)

# Iterate through epochs
for i in range(15):
    SRC_PATH = f"geometry_shadow_decoupling/arch_4/results/test_32_4_seg/epoch_{i}/chkpt.pt"
    checkpoint = torch.load(
        SRC_PATH,
        map_location="cpu",
        weights_only=False
    )

    # Gather epoch metrics
    mae_obj_seg.append(checkpoint["mae_obj"]*100)
    mae_bkg_seg.append(checkpoint["mae_bkg"]*100)
    mae_full_seg.append(checkpoint["mae"]*100)

# Display
plt.figure(0)
plt.title("Relative MAE on Depth")
plt.plot(mae_bkg, label="Background w/o")
plt.plot(mae_obj, label="RSO w/o")
plt.plot(mae_bkg_seg, linestyle='--', color='tab:blue', label="Background w")
plt.plot(mae_obj_seg, linestyle='--', color='tab:orange', label="RSO w")
plt.ylabel("Relative MAE [%]")
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=4
)
plt.tight_layout()

plt.show()

