# Import satellite abstraction
from sat_node import SatNode

# Parameters
from parameters import *

# Import standard modules
import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2
from datetime import datetime
import os
import time


# Training
def main(folder_name: str):
    
    # Instantiate the satellite node
    sat_node = SatNode(**cfg_parameters)

    # Rendez-vous around the target
    for i, (c2w, frame) in enumerate(sat_node.get_measurement()):

        # Save a test frame every N frames
        if i % TEST_EVERY == 0:

            sat_node.valid_set.append((c2w, frame))
            continue

        # Populate data buffer
        sat_node.train_set.get_data(c2w=c2w,
                                    frame=frame)
        
        # Visualize rendez-vous
        if DISP:
            cv2.imshow("Rendez-vous View", frame.detach().numpy())
            cv2.waitKey(1)

    
    
    # Close all visualizations, if any
    cv2.destroyAllWindows()

    # Create DataLoader object to shuffle rays
    dataloader = DataLoader(dataset=sat_node.train_set,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=2)
    
    # Auxiliary funcs
    psnr = lambda img1, img2, max_pixel_value = 1.0 : 10. * torch.log10(max_pixel_value ** 2 / torch.mean((img1 - img2) ** 2))

    # Start timing
    t_start = time.time()
    
    # Training loop
    for epoch in range(EPOCHS):
        
        # Iterate through rays samples
        for niter, ray_batch in enumerate(dataloader):

            # Train one batch
            loss_val = sat_node.trainer.train_one_batch(rays=ray_batch[:, :6],
                                                        labels=ray_batch[:, 6:])
            
            # Display loss values
            if VERB:
                print(f"Epoch: {epoch} | # Iter: {niter} | Elapsed time (s): {(time.time()-t_start):.3f} | Loss: {loss_val.item():.5f}")

        # Test model on test set
        with torch.no_grad():

            # Create test folder
            path_to_chkpt = f"{folder_name}/epoch_{epoch}/"
            path_for_frames = f"{path_to_chkpt}/frames/"
            os.makedirs(path_for_frames)
            
            # Create metrics buffer
            test_psnr_vals = []
            
            # Loop through the validation set
            for j, (test_c2w, test_frame) in enumerate(sat_node.valid_set):

                # Render frame and compute PSNR
                test_rendering, _, _ = sat_node.render(c2w=test_c2w)
                test_rendering = test_rendering.detach().cpu().numpy().reshape(sat_node.H, sat_node.W, 3)
                test_psnr_vals.append(psnr(test_frame, test_rendering))

                # Store rendering
                cv2.imwrite(f"{path_for_frames}/{j}.jpg", test_rendering * 255)

                # Store checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": sat_node.renderer.state_dict(),
                    "optimizer_state_dict": sat_node.trainer.optimizer.state_dict(),
                    "psnr": np.mean(test_psnr_vals)
                }
                torch.save(checkpoint, f"{path_to_chkpt}/chkpt.pth")
            
            # Print metrics
            if VERB:
                print(f"Testing after epoch {epoch} | PSNR: {np.mean(test_psnr_vals):.3f}\n\n")

        # Call scheduler at the end of each epoch
        sat_node.trainer.scheduler.step()


# Main
if __name__ == "__main__":

    # Create a folder for this training session
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"training_logs/folder_{current_time}"
    os.makedirs(folder_name)

    # Launch training loop
    main(folder_name=folder_name)