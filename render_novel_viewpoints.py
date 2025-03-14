# Standard import
import json
import numpy as np
import cv2

# Custom import
from parameters_synth import *
from generate_nerf_datasets import calculate_intrinsic_matrix
from rays_generator_synth import RaysGeneratorSynth
from rendering import NeuralRenderer

# Parameters
CALIB_PATH = "/home/visione/Projects/BlenderScenarios/Sat/Dataset/Orbit_V_256_dynlight/VIS/transforms.json"
MODEL_PATH = "training_logs/folder_2025-03-11_15-12-35/epoch_29/chkpt.pt"
WPS_PATH =   "data/test_wps.npy"


# Main function
def main():
    
    # Load calibration parameters
    with open(CALIB_PATH, "r") as fopen:
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
    )
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load waypoints and rescale distances
    # wps = np.load(WPS_PATH)[0]
    # wp = torch.from_numpy(wps.astype(np.float32)).reshape(4, 4)
    
    with torch.no_grad():
        
        # Load data for testing
        with open(dataset_parameters["test_path"] + "/transforms.json", "r") as train_fopen:
            test_df = json.load(train_fopen)
            wp = torch.tensor(test_df["frames"][0]["transform_matrix"])

        # Generate rays
        test_rays = raygen(wp).reshape(-1, 6)
        test_rays = test_rays.to(cfg_parameters["device"])

        while True:
            for angle in np.linspace(0, 2*np.pi, 50).tolist():
                
                # Change sun direction
                sd = torch.tensor([0, np.sin(angle), np.cos(angle)], dtype=torch.float32)

                # Generate sun direction
                test_sundir = sd.expand(test_rays.shape[0], -1)
                test_sundir = test_sundir.to(cfg_parameters["device"])
                
                # Estimate rendering
                test_rgb, test_depth, _, _, _ = model(test_rays, test_sundir)
                
                # Reconstruct images and store them
                frame = test_rgb.detach().cpu().numpy().reshape(cfg_parameters["resolution"], cfg_parameters["resolution"], cfg_parameters["channels"])

                # Display
                cv2.imshow("", frame)
                cv2.waitKey(10)


if __name__ == "__main__":
    main()