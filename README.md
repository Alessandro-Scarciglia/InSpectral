# Joint NeRF and Camera Pose Optimization Pipeline

This repository implements a robust, end-to-end framework for joint **Neural Radiance Fields (NeRF)** optimization and **learnable camera pose refinement** (similar to BARF). It includes full modules for dataset calibration, spatial trajectory centering, high-performance pixel-level preprocessing, multi-task training agents, and automated video generation.

## 📊 Pipeline Overview

The complete project workflow is divided into four main functional stages:

1. **Pre-Calibration & Centering:** Trajectory extraction, centering, and unit bounding-box scaling.
2. **Pixel Dataset Export:** Flat serialization of features into a single `.npy` file for high-throughput training.
3. **Multi-Task Training & Validation:** Joint optimization of scene radiance fields and camera extrinsic residuals.
4. **AVI Video Compilation:** Sequence parsing and automated video render generation.

---

## 📁 Repository Structure

* **`calibrate.py`** – Exploits the pretrained VGGT model to resolve initial unposed frames, aligns trajectories globally using the Umeyama algorithm, and automatically shifts/normalizes camera limits.
* **`preprocess_dataset.py`** – Generates flat, high-performance `.npy` dataset structures from images, masks, and poses for efficient ray-box training.
* **`trainer.py`** – A grouped-parameter optimization agent managing dynamic schedules (sparsity warm-up) and multi-task loss architectures.
* **`evaluate.py`** – Validates a trained model against test configurations using chunked volumetric rendering to safeguard VRAM.
* **`images_to_avi.py`** – Utility script to compile sequential rendering outputs (RGB, depth maps, or masks) into standard AVI clips.

---

## ⚙️ Detailed Module Breakdown

### 1. Dataset Pre-Calibration (`calibrate.py`)
This script takes an unposed frame folder sequence and prepares a calibrated structural JSON configuration file (`vggt_transforms.json`).

* **Geometric Trajectory Centering:** Calculates the exact bounding box midpoint of all camera translation centers:
  $$\text{Center}_{\text{scene}} = \frac{\min(\mathbf{T}) + \max(\mathbf{T})}{2}$$
* **Unit Box Scaling:** Isotropically scales the camera coordinate footprint to map positions cleanly within a tight unit bounding box volume of $[-1, 1]$.

### 2. High-Performance Dataset Preparation (`preprocess_dataset.py`)
Converts source image frames, camera matrices, and pixel masks into a continuous data array for training.

* **Storage Efficiency:** Avoids slow Python appending loops by allocating a continuous, flat NumPy block up-front.
* **Dynamic Tagging:** Appends the exact calculated path boundaries (`min_dist` and `max_dist`) directly into the saved file name (e.g., `Landsat_4_masked_arch_4_min1.23_max1.89.npy`).
* **Pixel Matrix Composition:** Generates $11$ distinct data channels per pixel array slice:
  $$\text{Pixel Feature Vector} = [R_o\ (3),\, R_d\ (3),\, \text{RGB}\ (3),\, \text{Mask}\ (1),\, \text{Frame ID}\ (1)]$$

### 3. Training Optimization Strategy (`trainer.py`)
The pipeline runs an end-to-end multi-task objective loss configuration managed by a unified training class.

* **Grouped Learning Rates:** Separates standard model networks, grid feature hash-mappers, appearance embeds, and residual camera estimators into individual learning blocks using `AdamW`.
* **Sparsity Linear Warm-up:** Implements a ramp schedule scaling during the initial $15\%$ of total execution iterations to prevent local minima collapses.
* **Multi-Task Optimization:** Dynamically structures standard Mean Squared Error (MSE) losses alongside Total Variation (TV) smoothness, semantic boundaries, and adjacent appearance embedding tracking:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{photometric}} + \lambda_{\text{sparsity}}\mathcal{L}_{\text{sparsity}} + \lambda_{\text{similarity}}\mathcal{L}_{\text{app\_sim}} + \lambda_{\text{tv}}\mathcal{L}_{\text{tv}} + \lambda_{\text{segm}}\mathcal{L}_{\text{BCE\_Dice}}$$

### 4. Volumetric Evaluation Pipeline (`evaluate.py`)
Calculates rigorous quantitative assessment statistics (Full/Object/Background PSNR and MAE) over test targets.
* **Chunked Spatial Inference:** Splits rendering rays into configurable micro-slices (e.g., $4096$ pixels) to avoid GPU memory overflow during test evaluation passes.

---

## 🚀 Getting Started

### 1. Configure the Framework
Adjust