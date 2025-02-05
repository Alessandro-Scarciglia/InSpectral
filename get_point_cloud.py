# Import modules
import torch
import open3d as o3d
import json
import numpy as np
import time

# Import custom modules
from generate_nerf_datasets import calculate_intrinsic_matrix
from rays_generator_synth import RaysGeneratorSynth
from rendering import NeuralRenderer
from parameters_synth import *


# Generate and store ground truth point cloud from CAD model
def pc_from_cad(src_path: str,
                dst_path: str,
                num_points: int = 10000):

    # Load .stl file
    mesh = o3d.io.read_triangle_mesh(src_path)

    # Sample points from the surface of the mesh
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    # Store the ground truth (it does not need to be computed dinamically)
    o3d.io.write_point_cloud(dst_path, pcd)

    return pcd


# Compute point cloud from NeRF model
def pc_from_model(model_path: str,
                  calib_path: str,
                  wps_path: str,
                  resolution: tuple = (256, 256),
                  sigma_th: float = 5,
                  alpha: float = 0.05):
    
    # Load calibration parameters
    with open(calib_path, "r") as fopen:
        fov = json.load(fopen)["camera_angle_x"]
    
    # Compute camera calibration
    K = calculate_intrinsic_matrix(fov=fov, resolution=resolution)

    # Define the rays generator object
    raygen = RaysGeneratorSynth(**rays_parameters, K=K)

    # Define the model
    model = NeuralRenderer(
        **sampler_parameters,
        **hash_parameters,
        **posenc_parameters,
        **sh_parameters,
        **nerf_parameters,
        device=cfg_parameters["device"]
    ).eval()

    # Load checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load waypoints and rescale distances
    wps = np.load(wps_path)

    # Disable gradient ret
    with torch.no_grad():

        # Compute inference on all waypoints and collect samples
        cloud_collector = torch.tensor([], dtype=torch.float32)
        for i, wp in enumerate(wps):

            # Transform in torch tensor
            wp = torch.from_numpy(wp.astype(np.float32)).reshape(4, 4)

            # Generate rays from waypoint
            rays = raygen(wp)
            rays = rays.reshape(-1, 6)

            # Inference (batching)
            samples = torch.tensor([], dtype=torch.float32, device=cfg_parameters["device"])
            sigmas = torch.tensor([], dtype=torch.float32, device=cfg_parameters["device"])
            for i in range(4):

                # Predict color-sigmas
                _, _, _, batch_samples, batch_raw = model(rays[i*(128**2): (i+1)*(128**2)])
                time.sleep(0.1)

                # Collect samples of reference (N x 3)
                batch_samples = batch_samples.reshape(-1, 3)
                samples = torch.cat([samples, batch_samples], dim=0)

                # Select sigmas only and reshape (N x 1)
                batch_sigmas = batch_raw[..., -1].reshape(-1)
                sigmas = torch.cat([sigmas, batch_sigmas], dim=0)
                
            # Filter out samples with zero sigma
            samples = samples[sigmas > sigma_th].cpu()
            sigmas = sigmas[sigmas > sigma_th].cpu()

            # Attach to cloud
            sample_sigma = torch.cat([samples, sigmas.unsqueeze(-1)], dim=-1)
            cloud_collector = torch.cat([cloud_collector, sample_sigma], dim=0)

        # Transform the cloud collector in numpy array to handle the plotting
        cloud_collector = cloud_collector.numpy()

        # Estrai le coordinate e le densità
        points = cloud_collector[:, :3]

        # Crea un oggetto PointCloud di Open3D
        pcd = o3d.geometry.PointCloud()

        # Aggiungi le coordinate al point cloud
        pcd.points = o3d.utility.Vector3dVector(points)

        # Calcolo della mesh concava (Alpha Shape)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        # Estrazione dei vertici della mesh come punti di superficie
        surface_points = np.asarray(mesh.vertices)

        # Creazione del nuovo point cloud
        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(surface_points)

        return surface_pcd
    

# Main
if __name__ == "__main__":

    # Create and load pc from stl
    pcd_gt = pc_from_cad(src_path="/home/visione/Projects/BlenderScenarios/Asteroid/1620geographos.stl",
                         dst_path="./gt.ply")
    
    # Compute pc from model
    pcd_est = pc_from_model(model_path="training_logs/folder_2025-02-05_09-21-39/epoch_19/chkpt.pt",
                            calib_path="/home/visione/Projects/BlenderScenarios/Asteroid/Dataset/Orbit_V_256/IR/transforms.json",
                            wps_path="data/test_wps.npy",
                            resolution=(256, 256),
                            sigma_th=5.,
                            alpha=.05)
    
    # Apply color to pcd1 (shades of blue)
    pcd_gt.paint_uniform_color([0, 0, 1])  # Solid blue

    # Apply color to pcd2 (shades of red)
    pcd_est.paint_uniform_color([1, 0, 0])  # Solid red

    # Visualize both point clouds together
    o3d.visualization.draw_geometries([pcd_est, pcd_gt], window_name="Point Clouds in Blue and Red")
