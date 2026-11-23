"""
This script transforms the images and labels of a custom NeRF dataset
into a ray-based NumPy dataset, where each sample corresponds to one pixel.

Optional features:
    - Temporally correlated radial position perturbations.
    - Temporally correlated attitude/pointing perturbations.
    - Noise is applied only to nominal trajectories.
    - Estimated trajectories are never perturbed.
    - Camera poses used for ray generation include both perturbations.
    - Final 3D visualization shows nominal and perturbed trajectories,
      including the camera reference frame at every waypoint.

Attitude convention:
    The camera-to-world rotation matrix is perturbed by right multiplication:

        R_noisy[k] = R_nominal[k] @ delta_R[k]

    Therefore, the perturbation is expressed in the local camera frame.

    Pointing errors are applied around the local camera X and Y axes.
    Rotation around the optical axis (local Z) is not perturbed.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from data.image_io import channels_for_mode, load_frame_image
from configs.config_parameters import IMAGE_MODE, CORRUPTED_POSES


# =======================================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =======================================================================================

CONFIG = {

    "target_resolution": 512,
    "src_dataset": "Landsat8_v4",
    "arch": "arch_4",
    "deg_res": 4,

    # Image configuration: 'rgb', 'ir' or 'rgb_ir' (RGB stacked with IR)
    "image_type": IMAGE_MODE,  # set IMAGE_MODE in configs/config_parameters.py

    # Relative path to the segmentation masks
    "masked": "images/mask_sam2_grounded_bi",

    # Trajectory selection: 'estimated' or 'nominal'
    "trajectory_type": "nominal",

    # -----------------------------------------------------------------------------------
    # POSITION PERTURBATION
    # -----------------------------------------------------------------------------------

    "enable_trajectory_noise": CORRUPTED_POSES,  # set CORRUPTED_POSES in configs/config_parameters.py

    # Relative standard deviation of radial position perturbations.
    # Example: 0.03 corresponds to 3%.
    "radial_noise_sigma": 0.03,

    # Temporal correlation coefficient between consecutive *kept* frames.
    "radial_noise_rho": 0.90,

    # Optional correlation length in degrees of orbit. When set, it replaces
    # radial_noise_rho with rho = exp(-frame_spacing_deg / corr_deg), so the physical
    # correlation stays the same whatever deg_res is. None keeps radial_noise_rho.
    "radial_noise_corr_deg": None,

    # -----------------------------------------------------------------------------------
    # ATTITUDE / POINTING PERTURBATION
    # -----------------------------------------------------------------------------------

    "enable_pointing_noise": CORRUPTED_POSES,

    # Standard deviation of the pointing error around each local camera axis.
    # The same standard deviation is applied independently to local X and Y.
    # Roll around local Z is not perturbed.
    "pointing_noise_sigma_deg": 0.20,

    # Temporal correlation coefficient between consecutive *kept* frames.
    "pointing_noise_rho": 0.90,

    # Optional correlation length in degrees of orbit (see radial_noise_corr_deg).
    "pointing_noise_corr_deg": None,

    # -----------------------------------------------------------------------------------
    # RANDOMNESS AND VISUALIZATION
    # -----------------------------------------------------------------------------------

    "noise_seed": 42,

    # Length of the camera reference-frame axes as a fraction
    # of the maximum trajectory dimension.
    "frame_axis_length_fraction": 0.06,

    # -----------------------------------------------------------------------------------
    # INPUT AND OUTPUT PATHS
    # -----------------------------------------------------------------------------------

    "data_path": "/home/vision/Desktop/Datasets/Landsat8_v4/",

    "data_dst_dir": (
        "/home/vision/Desktop/InSpectral/data/postprocessed_dataset/"
    )
}


# =======================================================================================
# CAMERA INTRINSIC MATRIX
# =======================================================================================

def calculate_intrinsic_matrix(
    fov: float,
    resolution: tuple[int, int]
) -> np.ndarray:

    """Calculate the intrinsic camera matrix for an ideal pinhole camera."""

    width, height = resolution

    fx = width / (2 * np.tan(fov / 2))
    fy = height / (2 * np.tan(fov / 2))

    cx = width / 2.0
    cy = height / 2.0

    return np.array([
        [fx,  0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])


# =======================================================================================
# FIRST-ORDER GAUSS-MARKOV NOISE GENERATION
# =======================================================================================

def generate_correlated_noise(
    num_samples: int,
    sigma: float,
    rho: float,
    rng: np.random.Generator
) -> np.ndarray:

    """
    Generate stationary, temporally correlated Gaussian noise.

    The first-order Gauss-Markov process is:

        x[k] = rho * x[k-1]
               + sigma * sqrt(1-rho^2) * z[k]

        z[k] ~ N(0, 1)

    The initial value is sampled from the stationary distribution:

        x[0] ~ N(0, sigma^2)

    Consequently, the process has zero mean and stationary variance sigma^2.
    """

    if sigma < 0:
        raise ValueError("Noise standard deviation must be non-negative.")

    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in the range [0, 1).")

    noise = np.zeros(num_samples, dtype=np.float64)

    if num_samples == 0:
        return noise

    noise[0] = rng.normal(0.0, sigma)

    for k in range(1, num_samples):

        noise[k] = (
            rho * noise[k - 1]
            + sigma * np.sqrt(1.0 - rho**2) * rng.normal()
        )

    return noise


# =======================================================================================
# RADIAL POSITION PERTURBATION
# =======================================================================================

def apply_radial_trajectory_noise(
    original_centers: np.ndarray,
    sigma: float,
    rho: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:

    """
    Apply temporally correlated Gaussian radial perturbations.

    The radial perturbation is:

        epsilon[k] = rho * epsilon[k-1]
                     + sigma * sqrt(1-rho^2) * z[k]

        z[k] ~ N(0, 1)

    The perturbed camera position is:

        p_noisy[k] = (1 + epsilon[k]) * p_nominal[k]

    This preserves the angular direction of each position vector while
    perturbing its radial distance from the coordinate-system origin.

    Args:
        original_centers: Nominal camera positions, shape (N, 3).
        sigma: Relative standard deviation of radial perturbations.
        rho: Temporal correlation coefficient.
        rng: Random number generator.

    Returns:
        noisy_centers: Perturbed camera positions, shape (N, 3).
        radial_noise: Realized relative radial perturbations, shape (N,).
    """

    if sigma < 0:
        raise ValueError("radial_noise_sigma must be non-negative.")

    if not 0.0 <= rho < 1.0:
        raise ValueError("radial_noise_rho must be in the range [0, 1).")

    num_frames = len(original_centers)

    radial_noise = generate_correlated_noise(
        num_samples=num_frames,
        sigma=sigma,
        rho=rho,
        rng=rng
    )

    # Prevent negative radial distances in the extremely unlikely
    # event that the Gaussian perturbation is smaller than -1.
    radial_scale = np.maximum(1.0 + radial_noise, 1e-3)

    noisy_centers = (
        original_centers * radial_scale[:, np.newaxis]
    )

    return noisy_centers, radial_noise


# =======================================================================================
# ROTATION MATRIX UTILITIES
# =======================================================================================

def skew_symmetric_matrix(vector: np.ndarray) -> np.ndarray:

    """Return the skew-symmetric matrix associated with a 3D vector."""

    x, y, z = vector

    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ])


def rotation_matrix_from_vector(rot_vec: np.ndarray) -> np.ndarray:

    """
    Convert a rotation vector into a rotation matrix using Rodrigues' formula.

    The rotation vector direction defines the rotation axis, while its
    magnitude defines the rotation angle in radians.
    """

    angle = np.linalg.norm(rot_vec)

    if angle < 1e-12:
        return np.eye(3)

    axis = rot_vec / angle

    K = skew_symmetric_matrix(axis)

    return (
        np.eye(3)
        + np.sin(angle) * K
        + (1.0 - np.cos(angle)) * (K @ K)
    )


# =======================================================================================
# ATTITUDE / POINTING PERTURBATION
# =======================================================================================

def apply_pointing_noise(
    nominal_poses: np.ndarray,
    sigma_deg: float,
    rho: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:

    """
    Apply temporally correlated pointing perturbations to camera poses.

    The perturbation is expressed in the local camera reference frame.

    Two independent first-order Gauss-Markov processes are generated:

        theta_x[k] = rho * theta_x[k-1]
                     + sigma * sqrt(1-rho^2) * z_x[k]

        theta_y[k] = rho * theta_y[k-1]
                     + sigma * sqrt(1-rho^2) * z_y[k]

    where:

        z_x, z_y ~ N(0, 1)

    The rotation vector is:

        theta[k] = [theta_x[k], theta_y[k], 0]^T

    The perturbed camera-to-world rotation matrix is:

        R_noisy[k] = R_nominal[k] @ delta_R[k]

    Thus, the perturbation is applied around the local camera X and Y axes.
    Rotation around local Z (roll) is not perturbed.

    Args:
        nominal_poses: Nominal camera-to-world matrices, shape (N, 4, 4).
        sigma_deg: Angular standard deviation, in degrees, for each
                   of the two perturbed axes.
        rho: Temporal correlation coefficient.
        rng: Random number generator.

    Returns:
        noisy_poses: Camera poses with perturbed orientations.
        angular_noise: Realized angular perturbations, shape (N, 2),
                       expressed in radians.
    """

    if sigma_deg < 0:
        raise ValueError(
            "pointing_noise_sigma_deg must be non-negative."
        )

    if not 0.0 <= rho < 1.0:
        raise ValueError("pointing_noise_rho must be in the range [0, 1).")

    num_frames = len(nominal_poses)

    sigma_rad = np.deg2rad(sigma_deg)

    # Independent noise processes for the two transverse camera axes.
    theta_x = generate_correlated_noise(
        num_samples=num_frames,
        sigma=sigma_rad,
        rho=rho,
        rng=rng
    )

    theta_y = generate_correlated_noise(
        num_samples=num_frames,
        sigma=sigma_rad,
        rho=rho,
        rng=rng
    )

    angular_noise = np.column_stack((theta_x, theta_y))

    noisy_poses = nominal_poses.copy()

    for k in range(num_frames):

        rot_vec = np.array([
            theta_x[k],
            theta_y[k],
            0.0
        ])

        delta_R = rotation_matrix_from_vector(rot_vec)

        R_nominal = nominal_poses[k, :3, :3]

        # Right multiplication: perturbation expressed in the local frame.
        noisy_poses[k, :3, :3] = R_nominal @ delta_R

    return noisy_poses, angular_noise


# =======================================================================================
# 3D TRAJECTORY VISUALIZATION UTILITIES
# =======================================================================================

def set_axes_equal_3d(ax, centers: np.ndarray):

    """Set equal scaling on all three axes of a 3D plot."""

    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    max_range = max(
        np.ptp(x),
        np.ptp(y),
        np.ptp(z)
    ) / 2.0

    max_range = max(max_range, 1e-6)

    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect((1, 1, 1))


def draw_camera_frames(
    ax,
    poses: np.ndarray,
    axis_length: float
):

    """
    Draw the local camera reference frame at every waypoint.

    The camera-to-world rotation matrix contains the camera axes expressed
    in world coordinates as its columns:

        R[:, 0] -> local camera X axis
        R[:, 1] -> local camera Y axis
        R[:, 2] -> local camera Z axis

    X: red
    Y: green
    Z: blue
    """

    for k, pose in enumerate(poses):

        origin = pose[:3, 3]
        R = pose[:3, :3]

        camera_x = R[:, 0]
        camera_y = R[:, 1]
        camera_z = R[:, 2]

        ax.quiver(
            origin[0], origin[1], origin[2],
            camera_x[0], camera_x[1], camera_x[2],
            length=axis_length,
            normalize=True,
            color='red',
            linewidth=1.0,
            arrow_length_ratio=0.25,
            label='Camera X' if k == 0 else None
        )

        ax.quiver(
            origin[0], origin[1], origin[2],
            camera_y[0], camera_y[1], camera_y[2],
            length=axis_length,
            normalize=True,
            color='green',
            linewidth=1.0,
            arrow_length_ratio=0.25,
            label='Camera Y' if k == 0 else None
        )

        ax.quiver(
            origin[0], origin[1], origin[2],
            camera_z[0], camera_z[1], camera_z[2],
            length=axis_length,
            normalize=True,
            color='blue',
            linewidth=1.0,
            arrow_length_ratio=0.25,
            label='Camera Z' if k == 0 else None
        )


# =======================================================================================
# TRAJECTORY VISUALIZATION
# =======================================================================================

def plot_trajectory(
    nominal_poses: np.ndarray,
    perturbed_poses: np.ndarray,
    normalized_nominal_poses: np.ndarray,
    normalized_perturbed_poses: np.ndarray,
    save_path: str,
    position_noise_applied: bool,
    pointing_noise_applied: bool,
    radial_noise: np.ndarray,
    angular_noise: np.ndarray
):

    """
    Generate a 3D visualization of nominal and perturbed camera trajectories.

    The figure contains:
        1. Original-scale nominal and perturbed trajectories.
        2. Normalized nominal and perturbed trajectories.

    The perturbed camera reference frame is drawn at every waypoint.
    """

    nominal_centers = nominal_poses[:, :3, 3]
    perturbed_centers = perturbed_poses[:, :3, 3]

    normalized_nominal_centers = (
        normalized_nominal_poses[:, :3, 3]
    )

    normalized_perturbed_centers = (
        normalized_perturbed_poses[:, :3, 3]
    )

    # Realized position and attitude errors.
    radial_rms = (
        np.sqrt(np.mean(radial_noise**2))
        if len(radial_noise) > 0 else 0.0
    )

    angular_magnitude = np.linalg.norm(
        angular_noise,
        axis=1
    )

    angular_rms_deg = (
        np.rad2deg(
            np.sqrt(np.mean(angular_magnitude**2))
        )
        if len(angular_magnitude) > 0 else 0.0
    )

    # Reference-frame axis length.
    trajectory_range = np.ptp(
        perturbed_centers,
        axis=0
    ).max()

    axis_length = max(
        trajectory_range * CONFIG["frame_axis_length_fraction"],
        1e-6
    )

    normalized_axis_length = (
        axis_length / max(
            np.linalg.norm(perturbed_centers, axis=1).max(),
            1e-6
        )
    )

    fig = plt.figure(figsize=(17, 8))

    # -----------------------------------------------------------------------------------
    # SUBPLOT 1: ORIGINAL SCALE
    # -----------------------------------------------------------------------------------

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # Nominal trajectory.
    ax1.plot(
        nominal_centers[:, 0],
        nominal_centers[:, 1],
        nominal_centers[:, 2],
        color='blue',
        linestyle='--',
        linewidth=1.5,
        label='Nominal trajectory'
    )

    # Perturbed trajectory.
    ax1.plot(
        perturbed_centers[:, 0],
        perturbed_centers[:, 1],
        perturbed_centers[:, 2],
        color='orange',
        linewidth=1.5,
        label='Perturbed trajectory'
    )

    # Waypoint markers.
    ax1.scatter(
        nominal_centers[:, 0],
        nominal_centers[:, 1],
        nominal_centers[:, 2],
        color='blue',
        s=12
    )

    ax1.scatter(
        perturbed_centers[:, 0],
        perturbed_centers[:, 1],
        perturbed_centers[:, 2],
        color='orange',
        s=12
    )

    # Draw the perturbed camera reference frame at every waypoint.
    draw_camera_frames(
        ax=ax1,
        poses=perturbed_poses,
        axis_length=axis_length
    )

    ax1.set_title("Original-Scale Camera Trajectory")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    set_axes_equal_3d(
        ax1,
        np.vstack((nominal_centers, perturbed_centers))
    )

    ax1.grid(True)

    # -----------------------------------------------------------------------------------
    # SUBPLOT 2: NORMALIZED SCALE
    # -----------------------------------------------------------------------------------

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax2.plot(
        normalized_nominal_centers[:, 0],
        normalized_nominal_centers[:, 1],
        normalized_nominal_centers[:, 2],
        color='blue',
        linestyle='--',
        linewidth=1.5,
        label='Nominal trajectory'
    )

    ax2.plot(
        normalized_perturbed_centers[:, 0],
        normalized_perturbed_centers[:, 1],
        normalized_perturbed_centers[:, 2],
        color='orange',
        linewidth=1.5,
        label='Perturbed trajectory'
    )

    ax2.scatter(
        normalized_nominal_centers[:, 0],
        normalized_nominal_centers[:, 1],
        normalized_nominal_centers[:, 2],
        color='blue',
        s=12
    )

    ax2.scatter(
        normalized_perturbed_centers[:, 0],
        normalized_perturbed_centers[:, 1],
        normalized_perturbed_centers[:, 2],
        color='orange',
        s=12
    )

    # Draw the perturbed camera reference frame at every normalized waypoint.
    draw_camera_frames(
        ax=ax2,
        poses=normalized_perturbed_poses,
        axis_length=normalized_axis_length
    )

    ax2.set_title("Normalized Camera Trajectory")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    set_axes_equal_3d(
        ax2,
        np.vstack((
            normalized_nominal_centers,
            normalized_perturbed_centers
        ))
    )

    ax2.grid(True)

    # -----------------------------------------------------------------------------------
    # GLOBAL TITLE: APPLIED POSITION AND ATTITUDE ERRORS
    # -----------------------------------------------------------------------------------

    if position_noise_applied:
        position_label = (
            f"Radial position noise: "
            f"{CONFIG['radial_noise_sigma'] * 100:.2f}% "
            f"(rho={CONFIG['radial_noise_rho']:.2f})"
        )
    else:
        position_label = "Radial position noise: OFF"

    if pointing_noise_applied:
        pointing_label = (
            f"Pointing noise: "
            f"{CONFIG['pointing_noise_sigma_deg']:.3f} deg per axis "
            f"(rho={CONFIG['pointing_noise_rho']:.2f})"
        )
    else:
        pointing_label = "Pointing noise: OFF"

    title = (
        f"Camera Trajectory - {CONFIG['trajectory_type'].upper()}\n"
        f"{position_label} | {pointing_label}\n"
        f"Realized radial RMS: {radial_rms * 100:.3f}% | "
        f"Realized pointing RMS: {angular_rms_deg:.4f} deg"
    )

    fig.suptitle(
        title,
        fontsize=12,
        fontweight='bold'
    )

    # -----------------------------------------------------------------------------------
    # LEGEND
    # -----------------------------------------------------------------------------------

    trajectory_legend = [
        Line2D(
            [0], [0],
            color='blue',
            linestyle='--',
            label='Nominal trajectory'
        ),
        Line2D(
            [0], [0],
            color='orange',
            label='Perturbed trajectory'
        ),
        Line2D(
            [0], [0],
            color='red',
            label='Camera X'
        ),
        Line2D(
            [0], [0],
            color='green',
            label='Camera Y'
        ),
        Line2D(
            [0], [0],
            color='blue',
            label='Camera Z'
        )
    ]

    fig.legend(
        handles=trajectory_legend,
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.88])

    plt.savefig(
        save_path,
        dpi=250,
        bbox_inches='tight'
    )

    print(
        f"[INFO] Trajectory plot successfully saved to: {save_path}"
    )

    plt.close()


# =======================================================================================
# MAIN DATASET GENERATION PIPELINE
# =======================================================================================

def main():

    """
    Pipeline to produce and store postprocessed datasets,
    converting NeRF images and metadata into ray-based samples.
    """

    # -----------------------------------------------------------------------------------
    # CONFIGURATION VALIDATION
    # -----------------------------------------------------------------------------------

    if CONFIG["deg_res"] <= 0:
        raise ValueError("deg_res must be a positive integer.")

    if CONFIG["target_resolution"] <= 0:
        raise ValueError("target_resolution must be positive.")

    # -----------------------------------------------------------------------------------
    # DYNAMIC METADATA PATH SELECTION
    # -----------------------------------------------------------------------------------

    if CONFIG["trajectory_type"] == "estimated":
        json_filename = "transforms_est.json"

    elif CONFIG["trajectory_type"] == "nominal":
        json_filename = "transforms_train.json"

    else:
        raise ValueError(
            f"Unknown trajectory_type: '{CONFIG['trajectory_type']}'. "
            "Choose 'estimated' or 'nominal'."
        )

    json_path = os.path.join(
        CONFIG["data_path"],
        json_filename
    )

    print(f"[INFO] Loading metadata from: {json_path}")

    with open(json_path, "r") as fopen:

        df = json.load(fopen)

        fov = df["camera_angle_x"]
        num_ch = int(df["num_ch"])
        all_samples = df["frames"]

    resolution = CONFIG["target_resolution"]

    samples = all_samples[::CONFIG["deg_res"]]

    num_frames = len(samples)

    if num_frames == 0:
        raise ValueError("No frames found in the selected dataset.")

    pixels_per_frame = resolution * resolution
    total_pixels = num_frames * pixels_per_frame

    print(
        f"[INFO] Processing {num_frames} frames out of "
        f"{len(all_samples)} total frames at resolution "
        f"{resolution}x{resolution}."
    )

    # -----------------------------------------------------------------------------------
    # EXTRACT ORIGINAL CAMERA POSES
    # -----------------------------------------------------------------------------------

    nominal_poses = np.array(
        [sample["transform_matrix"] for sample in samples],
        dtype=np.float64
    )

    if nominal_poses.shape[1:] != (4, 4):
        raise ValueError(
            "Expected camera-to-world matrices with shape (4, 4)."
        )

    nominal_centers = nominal_poses[:, :3, 3].copy()

    # Initialize perturbed poses and trajectory.
    perturbed_poses = nominal_poses.copy()

    trajectory_centers = nominal_centers.copy()

    # Noise diagnostics.
    radial_noise = np.zeros(num_frames, dtype=np.float64)

    angular_noise = np.zeros(
        (num_frames, 2),
        dtype=np.float64
    )

    position_noise_applied = False
    pointing_noise_applied = False

    # -----------------------------------------------------------------------------------
    # RANDOM NUMBER GENERATORS
    # -----------------------------------------------------------------------------------

    # Independent random streams ensure that enabling/disabling one noise
    # source does not change the realization of the other noise source.
    # -----------------------------------------------------------------------------------
    # NOISE CORRELATION (PER FRAME vs PER DEGREE OF ORBIT)
    # -----------------------------------------------------------------------------------

    # Angular spacing between consecutive kept frames, seen from the target.
    unit_centers = nominal_centers / np.linalg.norm(nominal_centers, axis=1, keepdims=True)
    frame_spacing_deg = float(np.rad2deg(np.mean(np.arccos(np.clip(
        np.sum(unit_centers[1:] * unit_centers[:-1], axis=1), -1.0, 1.0
    ))))) if num_frames > 1 else 0.0

    for prefix in ("radial_noise", "pointing_noise"):
        corr_deg = CONFIG.get(f"{prefix}_corr_deg")

        if corr_deg is not None:
            if corr_deg <= 0:
                raise ValueError(f"{prefix}_corr_deg must be positive.")
            CONFIG[f"{prefix}_rho"] = float(np.exp(-frame_spacing_deg / corr_deg))

        rho = CONFIG[f"{prefix}_rho"]
        equivalent = (
            f"{-frame_spacing_deg / np.log(rho):.1f} deg of orbit"
            if 0.0 < rho < 1.0 else ("none" if rho == 0.0 else "infinite")
        )
        print(
            f"[INFO] {prefix}: rho={rho:.3f} per kept frame "
            f"(frame spacing {frame_spacing_deg:.2f} deg, correlation length {equivalent})"
        )

    seed_sequence = np.random.SeedSequence(CONFIG["noise_seed"])

    position_seed, attitude_seed = seed_sequence.spawn(2)

    position_rng = np.random.default_rng(position_seed)
    attitude_rng = np.random.default_rng(attitude_seed)

    # -----------------------------------------------------------------------------------
    # OPTIONAL POSITION AND ATTITUDE NOISE
    # NOMINAL TRAJECTORIES ONLY
    # -----------------------------------------------------------------------------------

    if CONFIG["trajectory_type"] == "nominal":

        # -------------------------------------------------------------------------------
        # RADIAL POSITION NOISE
        # -------------------------------------------------------------------------------

        if CONFIG["enable_trajectory_noise"]:

            trajectory_centers, radial_noise = (
                apply_radial_trajectory_noise(
                    original_centers=nominal_centers,
                    sigma=CONFIG["radial_noise_sigma"],
                    rho=CONFIG["radial_noise_rho"],
                    rng=position_rng
                )
            )

            perturbed_poses[:, :3, 3] = trajectory_centers

            position_noise_applied = True

            print(
                "[INFO] Radial position noise enabled: "
                f"sigma={CONFIG['radial_noise_sigma']:.4f}, "
                f"rho={CONFIG['radial_noise_rho']:.2f}"
            )

        # -------------------------------------------------------------------------------
        # ATTITUDE / POINTING NOISE
        # -------------------------------------------------------------------------------

        if CONFIG["enable_pointing_noise"]:

            perturbed_poses, angular_noise = apply_pointing_noise(
                nominal_poses=perturbed_poses,
                sigma_deg=CONFIG["pointing_noise_sigma_deg"],
                rho=CONFIG["pointing_noise_rho"],
                rng=attitude_rng
            )

            pointing_noise_applied = True

            print(
                "[INFO] Pointing noise enabled: "
                f"sigma={CONFIG['pointing_noise_sigma_deg']:.4f} deg "
                f"per local axis, "
                f"rho={CONFIG['pointing_noise_rho']:.2f}"
            )

    else:

        print(
            "[INFO] Estimated trajectory selected: "
            "position and attitude noise disabled."
        )

    # -----------------------------------------------------------------------------------
    # PERTURBATION STATISTICS
    # -----------------------------------------------------------------------------------

    radial_rms = (
        np.sqrt(np.mean(radial_noise**2))
        if position_noise_applied else 0.0
    )

    angular_magnitude = np.linalg.norm(
        angular_noise,
        axis=1
    )

    pointing_rms_deg = (
        np.rad2deg(
            np.sqrt(np.mean(angular_magnitude**2))
        )
        if pointing_noise_applied else 0.0
    )

    position_displacement = np.linalg.norm(
        perturbed_poses[:, :3, 3] - nominal_centers,
        axis=1
    )

    position_displacement_rms = np.sqrt(
        np.mean(position_displacement**2)
    )

    print("\n[INFO] Applied perturbation statistics")
    print("---------------------------------------")

    print(
        f"Radial perturbation RMS: {radial_rms * 100:.4f}%"
    )

    print(
        f"Position displacement RMS: "
        f"{position_displacement_rms:.6f} "
        f"(input coordinate units)"
    )

    print(
        f"Pointing perturbation RMS: {pointing_rms_deg:.6f} deg"
    )

    print("---------------------------------------\n")

    # -----------------------------------------------------------------------------------
    # TRAJECTORY NORMALIZATION
    # -----------------------------------------------------------------------------------

    distances = np.linalg.norm(
        trajectory_centers,
        axis=1
    )

    min_dist = float(distances.min())
    max_dist = float(distances.max())

    scale_factor = 1.0 / (max_dist + 1e-10)

    print(
        f"[INFO] Trajectory bounds from origin - "
        f"Min Dist: {min_dist:.4f}, "
        f"Max Dist: {max_dist:.4f}"
    )

    normalized_nominal_poses = nominal_poses.copy()
    normalized_perturbed_poses = perturbed_poses.copy()

    normalized_nominal_poses[:, :3, 3] *= scale_factor

    normalized_perturbed_poses[:, :3, 3] *= scale_factor

    normalized_centers = (
        normalized_perturbed_poses[:, :3, 3]
    )

    # -----------------------------------------------------------------------------------
    # IMAGE CONFIGURATION (RGB VS IR)
    # -----------------------------------------------------------------------------------

    image_mode = CONFIG["image_type"].lower()
    img_channels = channels_for_mode(image_mode)

    print(
        f"[INFO] Configured mode: {image_mode.upper()} "
        f"({img_channels} channel(s))"
    )

    # -----------------------------------------------------------------------------------
    # MODEL AND FILENAME SETUP
    # -----------------------------------------------------------------------------------

    intrinsic_matrix = calculate_intrinsic_matrix(
        fov=fov,
        resolution=(resolution, resolution)
    )

    raygen = RaysGeneratorSynth(
        height=resolution,
        width=resolution,
        num_ch=num_ch,
        intrinsic_matrix=intrinsic_matrix
    )

    safe_masked_name = CONFIG["masked"].replace("/", "_")

    noise_suffix = ""

    if position_noise_applied:

        noise_suffix += (
            f"_radial{CONFIG['radial_noise_sigma']:.3f}"
            f"_rhor{CONFIG['radial_noise_rho']:.2f}"
        )

    if pointing_noise_applied:

        noise_suffix += (
            f"_point{CONFIG['pointing_noise_sigma_deg']:.3f}"
            f"_rhop{CONFIG['pointing_noise_rho']:.2f}"
        )

    base_name = (
        f"{CONFIG['src_dataset']}_"
        f"{CONFIG['deg_res']}_"
        f"{CONFIG['trajectory_type']}_"
        f"{safe_masked_name}_"
        f"{CONFIG['arch']}_"
        f"res{resolution}_"
        f"min{min_dist:.2f}_"
        f"max{max_dist:.2f}_"
        f"{image_mode}"
        f"{noise_suffix}"
    )

    filename = f"{base_name}.npy"

    plot_filename = f"{base_name}_trajectory.png"

    os.makedirs(
        CONFIG["data_dst_dir"],
        exist_ok=True
    )

    data_dst = os.path.join(
        CONFIG["data_dst_dir"],
        filename
    )

    plot_dst = os.path.join(
        CONFIG["data_dst_dir"],
        plot_filename
    )

    # -----------------------------------------------------------------------------------
    # DATASET ALLOCATION
    # -----------------------------------------------------------------------------------

    # Columns:
    # 6 (rays) + image channels + 1 (mask) + 1 (image index)

    total_features = 6 + img_channels + 1 + 1

    dataset = np.zeros(
        (total_pixels, total_features),
        dtype=np.float32
    )

    # -----------------------------------------------------------------------------------
    # VECTORIZED DATASET GENERATION
    # -----------------------------------------------------------------------------------

    print("Generating Training Dataset...")

    for idx, sample in enumerate(tqdm(samples)):

        # -------------------------------------------------------------------------------
        # LOAD IMAGE
        # -------------------------------------------------------------------------------

        img = load_frame_image(
            dataset_root=CONFIG["data_path"],
            frame=sample,
            mode=image_mode,
            resolution=resolution
        )

        # -------------------------------------------------------------------------------
        # LOAD SEGMENTATION MASK
        # -------------------------------------------------------------------------------

        mask_filename = os.path.basename(
            sample["mask_path"]
        )

        dynamic_mask_path = os.path.join(
            CONFIG["masked"],
            mask_filename
        )

        mask_path = os.path.join(
            CONFIG["data_path"],
            dynamic_mask_path
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise FileNotFoundError(
                f"[ERROR] Could not load mask: {mask_path}"
            )

        mask = cv2.resize(
            mask,
            (resolution, resolution)
        )

        # -------------------------------------------------------------------------------
        # APPLY SEGMENTATION MASK TO IMAGE
        # -------------------------------------------------------------------------------

        img = cv2.bitwise_and(
            img,
            img,
            mask=mask
        )

        if img_channels == 1 and len(img.shape) == 2:
            img = img[:, :, np.newaxis]

        # -------------------------------------------------------------------------------
        # NORMALIZE IMAGE AND MASK
        # -------------------------------------------------------------------------------

        img_norm = (
            img.astype(np.float32) / 255.0
        )

        mask_norm = (
            mask.astype(np.float32)[:, :, np.newaxis] / 255.0
        )

        # -------------------------------------------------------------------------------
        # CAMERA POSE AND RAY GENERATION
        # -------------------------------------------------------------------------------

        # Use the perturbed camera-to-world matrix.
        # Both translation and rotation include the applied perturbations.

        c2w = normalized_perturbed_poses[idx].astype(
            np.float32
        )

        # Generate rays using the perturbed and normalized camera pose.
        rays = raygen(
            torch.tensor(c2w)
        ).numpy()

        # -------------------------------------------------------------------------------
        # IMAGE INDEX GRID
        # -------------------------------------------------------------------------------

        image_index = np.full(
            (resolution, resolution, 1),
            idx,
            dtype=np.float32
        )

        # -------------------------------------------------------------------------------
        # CONCATENATE RAYS, IMAGE, MASK AND IMAGE INDEX
        # -------------------------------------------------------------------------------

        frame_data = np.concatenate(
            [
                rays,
                img_norm,
                mask_norm,
                image_index
            ],
            axis=-1
        )

        frame_data = frame_data.reshape(
            -1,
            total_features
        )

        # -------------------------------------------------------------------------------
        # STORE IN PREALLOCATED DATASET
        # -------------------------------------------------------------------------------

        start_idx = idx * pixels_per_frame
        end_idx = start_idx + pixels_per_frame

        dataset[start_idx:end_idx] = frame_data

    # -----------------------------------------------------------------------------------
    # TRAJECTORY VISUALIZATION
    # -----------------------------------------------------------------------------------

    plot_trajectory(
        nominal_poses=nominal_poses,
        perturbed_poses=perturbed_poses,
        normalized_nominal_poses=normalized_nominal_poses,
        normalized_perturbed_poses=normalized_perturbed_poses,
        save_path=plot_dst,
        position_noise_applied=position_noise_applied,
        pointing_noise_applied=pointing_noise_applied,
        radial_noise=radial_noise,
        angular_noise=angular_noise
    )

    # -----------------------------------------------------------------------------------
    # SHUFFLE AND SAVE
    # -----------------------------------------------------------------------------------

    print(
        f"Shuffling and saving {num_frames} processed frames..."
    )

    # Reproducible dataset shuffling.
    shuffle_rng = np.random.default_rng(
        CONFIG["noise_seed"] + 1
    )

    shuffle_rng.shuffle(dataset)

    np.save(
        data_dst,
        dataset
    )

    print(
        f"Dataset successfully saved to: {data_dst} | "
        f"Final Shape: {dataset.shape}"
    )

    # -----------------------------------------------------------------------------------
    # CAMERA POSE SIDECAR
    # -----------------------------------------------------------------------------------

    # The training table only stores per-pixel rays, so the camera poses used to build it
    # are saved next to it. Training uses them for the optional trajectory-smoothness
    # prior and, when ground-truth poses are known, to report how far the learned pose
    # corrections are from the truth.
    ground_truth_poses = normalized_nominal_poses

    if CONFIG["trajectory_type"] == "estimated":
        ground_truth_poses = None
        gt_json_path = os.path.join(CONFIG["data_path"], "transforms_train.json")

        if os.path.exists(gt_json_path):
            with open(gt_json_path, "r") as fopen:
                gt_frames = json.load(fopen)["frames"][::CONFIG["deg_res"]]

            if len(gt_frames) == num_frames:
                ground_truth_poses = np.array(
                    [frame["transform_matrix"] for frame in gt_frames],
                    dtype=np.float64
                )
                ground_truth_poses[:, :3, 3] *= scale_factor

    poses_dst = os.path.join(
        CONFIG["data_dst_dir"],
        f"{base_name}_poses.npz"
    )

    np.savez(
        poses_dst,
        input_poses=normalized_perturbed_poses.astype(np.float32),
        ground_truth_poses=(
            ground_truth_poses.astype(np.float32)
            if ground_truth_poses is not None
            else np.zeros((0, 4, 4), dtype=np.float32)
        ),
        scale_factor=np.float64(scale_factor)
    )

    print(f"Camera poses saved to: {poses_dst}")


# =======================================================================================
# SCRIPT ENTRY POINT
# =======================================================================================

if __name__ == "__main__":

    from data.rays_generator import RaysGeneratorSynth

    main()