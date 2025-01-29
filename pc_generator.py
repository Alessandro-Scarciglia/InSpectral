# Import modules
import torch
import json
import numpy as np
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
import time

# Import custom modules
from rays_generator_synth import RaysGeneratorSynth
from rendering import NeuralRenderer
from parameters_synth import *
from generate_nerf_datasets import calculate_intrinsic_matrix

# Main function
def main():

    # Load calibration parameters
    with open("/home/visione/Projects/BlenderScenarios/Asteroid/Dataset/Orbit_V_256/IR/transforms.json", "r") as fopen:
        fov = json.load(fopen)["camera_angle_x"]

    # Compute camera calibration
    K = calculate_intrinsic_matrix(fov=fov, resolution=(256, 256))

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
    checkpoint = torch.load("training_logs/folder_2025-01-28_15-20-45/epoch_0/chkpt.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load waypoints and rescale distances
    wps = np.load("data/test_wps.npy")

    # Disable gradient ret
    with torch.no_grad():

        # Compute inference on all waypoints and collect samples
        cloud_collector = torch.tensor([], dtype=torch.float32)
        for i, wp in tqdm(enumerate(wps)):

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
            samples = samples[sigmas > 3].cpu()
            sigmas = sigmas[sigmas > 3].cpu()

            # Attach to cloud
            sample_sigma = torch.cat([samples, sigmas.unsqueeze(-1)], dim=-1)
            cloud_collector = torch.cat([cloud_collector, sample_sigma], dim=0)

        # Transform the cloud collector in numpy array to handle the plotting
        cloud_collector = cloud_collector.numpy()

        # Estrai le coordinate e le densità
        points = cloud_collector[:, :3]  # Coordinate (x, y, z)
        density = cloud_collector[:, 3]  # Densità

        # Crea un oggetto PointCloud di Open3D
        pcd = o3d.geometry.PointCloud()

        # Aggiungi le coordinate al point cloud
        pcd.points = o3d.utility.Vector3dVector(points)

        # Normalizza la densità per utilizzarla come trasparenza
        norm_density = density / np.max(density)

        # Crea i colori in base alla densità (usando una mappa di colori)
        colors = plt.cm.viridis(norm_density)[:, :3]  # Usa la colormap 'viridis'

        # Aggiungi i colori al point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualizza il point cloud
        o3d.visualization.draw_geometries([pcd])
        

if __name__ == "__main__":
    main()

